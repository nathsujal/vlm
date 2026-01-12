import torch
from PIL import Image
import numpy as np
import cv2
from typing import Union, List, Dict, Optional, Tuple, Any
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from collections import defaultdict, deque
import hashlib
from datetime import datetime

from src.utils import get_logger, retry

logger = get_logger(__name__)


class BLIPEngine:
    """
    BLIP engine for image captioning and visual question answering.
    """

    def __init__(
        self,
        caption_model: str = "Salesforce/blip-image-captioning-large",
        vqa_model: str = "Salesforce/blip-vqa-base",
        device: str = "cpu",
        batch_size: int = 8,
    ):
        """
        Initialize BLIP Engine.
        
        Args:
            caption_model: HuggingFace model for image captioning
            vqa_model: HuggingFace model for visual question answering
            device: Device to run models on (cuda/cpu)
            batch_size: Batch size for processing multiple objects
        """
        self.device = device
        self.batch_size = batch_size
        
        # Stats
        self.caption_count = 0
        self.vqa_count = 0
        
        try:
            self._initialize_captioning_model(caption_model)
        except Exception as e:
            logger.error(f"Failed to initialize {caption_model}: {e}")
            raise RuntimeError(f"BLIP Engine initialization failed: {e}") from e
        try:
            self._initialize_vqa_model(vqa_model)
        except Exception as e:
            logger.error(f"Failed to initialize {vqa_model}: {e}")
            raise RuntimeError(f"BLIP Engine initialization failed: {e}") from e

        logger.info(
            f"BLIP Engine initialized | Device: {device}"
        )

    def _initialize_captioning_model(self, caption_model: str = "Salesforce/blip-image-captioning-large"):
        """
        Initialize captioning model.
        """
        logger.info(f"Loading captioning model: {caption_model}")
        self.caption_processor = BlipProcessor.from_pretrained(caption_model)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(caption_model)
        self.caption_model.to(self.device)
        self.caption_model.eval()

    def _initialize_vqa_model(self, vqa_model: str = "Salesforce/blip-vqa-base"):
        """
        Initialize VQA model.
        """
        logger.info(f"Loading VQA model: {vqa_model}")
        self.vqa_processor = BlipProcessor.from_pretrained(vqa_model)
        self.vqa_model = BlipForQuestionAnswering.from_pretrained(vqa_model)
        self.vqa_model.to(self.device)
        self.vqa_model.eval()

    def _preprocess_image(
        self,
        image: Union[str, np.ndarray, Image.Image],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        expand_ratio: float = 0.05
    ) -> Image.Image:
        """
        Preprocess image with optional cropping and context preservation.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            bbox: Optional bounding box (x1, y1, x2, y2) to crop
            expand_ratio: Ratio to expand bbox to preserve context
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to PIL Image
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # OpenCV uses BGR, convert to RGB
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Crop to bbox if provided
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            w, h = pil_image.size
            
            # Expand bbox to preserve context
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            expand_w = int(bbox_w * expand_ratio)
            expand_h = int(bbox_h * expand_ratio)
            
            x1 = int(max(0, x1 - expand_w))
            y1 = int(max(0, y1 - expand_h))
            x2 = int(min(w, x2 + expand_w))
            y2 = int(min(h, y2 + expand_h))
            
            pil_image = pil_image.crop((x1, y1, x2, y2))
            
        return pil_image

    @retry(retries=3, delay=1.0, backoff=2.0)
    def caption_image(
        self,
        image: Union[str, np.ndarray, Image.Image],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        conditional_prompt: Optional[str] = None,
        max_length: int = 30,
        num_beams: int = 3,
    ) -> str:
        """
        Generate caption for image or image region.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            bbox: Optional bounding box (x1, y1, x2, y2) to crop
            conditional_prompt: Optional text prompt to guide generation
            max_length: Maximum caption length in tokens
            num_beams: Beam search width (higher = better quality, slower)
            
        Returns:
            Generated caption string
        """
        # Preprocess image
        pil_image = self._preprocess_image(image, bbox)
        
        # Prepare inputs
        if conditional_prompt:
            inputs = self.caption_processor(
                pil_image,
                text=conditional_prompt,
                return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.caption_processor(
                pil_image,
                return_tensors="pt"
            ).to(self.device)
        
        # Generate caption
        with torch.no_grad():
            outputs = self.caption_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode caption
        caption = self.caption_processor.decode(
            outputs.sequences[0],
            skip_special_tokens=True
        )
        
        self.caption_count += 1
        return caption

    @retry(retries=3, delay=1.0, backoff=2.0)
    def answer_question(
        self,
        image: Union[str, np.ndarray, Image.Image],
        question: str,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        max_length: int = 20,
        num_beams: int = 3,
    ) -> str:
        """
        Answer a question about the image using VQA.
        
        Args:
            image: Input image
            question: Question to answer
            bbox: Optional bounding box to focus on
            max_length: Maximum answer length
            num_beams: Beam search width
            
        Returns:
            Answer string
        """
        # Preprocess image
        pil_image = self._preprocess_image(image, bbox)
        
        # Prepare inputs
        inputs = self.vqa_processor(
            pil_image,
            question,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.vqa_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode answer
        answer = self.vqa_processor.decode(
            outputs.sequences[0],
            skip_special_tokens=True
        )
    
        self.vqa_count += 1
        return answer

    @retry(retries=3, delay=1.0, backoff=2.0)
    def batch_caption_image(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
        max_length: int = 50,
        num_beams: int = 3,
    ) -> List[str]:
        """
        Batch process multiple images efficiently.
        
        Args:
            images: List of images
            bboxes: Optional list of bounding boxes (one per image)
            max_length: Maximum caption length
            
        Returns:
            List of caption dictionaries
        """
        if bboxes is None:
            bboxes = [None] * len(images)
        
        captions = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            batch_bboxes = bboxes[i:i + self.batch_size]
            
            # Preprocess batch
            pil_images = [
                self._preprocess_image(img, bbox)
                for img, bbox in zip(batch_images, batch_bboxes)
            ]
            
            # Prepare batch inputs
            inputs = self.caption_processor(
                pil_images,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Generate captions
            with torch.no_grad():
                outputs = self.caption_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode batch
            for j, seq in enumerate(outputs.sequences):
                caption = self.caption_processor.decode(seq, skip_special_tokens=True)
                
                captions.append(caption)
                
                self.caption_count += 1
        
        return captions


    def get_stats(self) -> dict:
        """Get engine statistics."""
        return {
            "captions_generated": self.caption_count,
            "vqa_queries": self.vqa_count
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.caption_count = 0
        self.vqa_count = 0
        logger.info("Statistics reset")

    def reset(self):
        """Reset all state (for new video)."""
        self.caption_count = 0
        self.vqa_count = 0
        logger.info("BLIP engine reset")