import clip
import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Optional

from src.models import Frame
from src.utils import get_logger, retry

logger = get_logger(__name__)


class CLIPEngine:
    """CLIP model wrapper."""
    
    def __init__(
        self, 
        model_name: str = "ViT-B/32", 
        device: str = "cpu",
        max_batch_size: int = 8
    ):
        """
        Initialize CLIP engine.
        
        Args:
            model_name: CLIP model variant
            device: Device for inference (cuda/cpu)
            max_batch_size: Maximum batch size for processing
        """
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device)
        self.device = device
        self.model.eval()

        # Get embedding dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(self.device)
            self.embedding_dim = self.model.encode_image(dummy).shape[-1]

        # Batch config
        self.max_batch_size = max_batch_size

        logger.info(
            f"CLIPEngine initialized | Model: {model_name} | "
            f"Device: {device} | Dim: {self.embedding_dim}"
        )
    
    @retry(retries=3, delay=1.0, backoff=2.0)
    def encode_frame(self, frame: Frame) -> np.ndarray:
        """
        Encode frame to embedding.
        
        Args:
            frame: Frame to encode
        
        Returns:
            Embedding array (512-dim for ViT-B/32)
        """
        return self._compute_embedding(frame.image)
    
    @retry(retries=3, delay=1.0, backoff=2.0)
    def batch_encode_frames(self, frames: List[Frame]) -> List[np.ndarray]:
        """
        Batch encode frames.
        
        Args:
            frames: List of frames to encode
        
        Returns:
            List of embeddings (one per frame)
        """
        if not frames:
            return []
        
        return self._batch_compute(frames)
    
    def _compute_embedding(self, image: np.ndarray) -> np.ndarray:
        """Compute single embedding (internal)."""
        image_tensor = self.preprocess(
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.cpu().numpy()[0]
    
    def _batch_compute(self, frames: List[Frame]) -> np.ndarray:
        """Batch compute embeddings with chunking for stability."""
        # Chunk large batches
        if len(frames) > self.max_batch_size:
            logger.debug(
                f"Chunking {len(frames)} frames into batches of {self.max_batch_size}"
            )
            chunks = []
            for i in range(0, len(frames), self.max_batch_size):
                chunk = frames[i:i + self.max_batch_size]
                chunk_embeddings = self._batch_compute(chunk)
                chunks.append(chunk_embeddings)
            return np.vstack(chunks)
        
        # Process batch
        images = [
            self.preprocess(
                Image.fromarray(cv2.cvtColor(f.image, cv2.COLOR_BGR2RGB))
            ) for f in frames
        ]
        
        batch_tensor = torch.stack(images).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.encode_image(batch_tensor)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings.cpu().numpy()
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity."""
        return float(
            np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        )
    
    def get_stats(self) -> dict:
        """Get engine statistics."""
        return {
            "embedding_dim": self.embedding_dim,
            "max_batch_size": self.max_batch_size
        }