import numpy as np
import cv2
from typing import Optional, List

from src.vlm import BLIPEngine
from src.models import Frame, Object
from src.utils import get_logger

logger = get_logger(__name__)

# Default caption parameters
DEFAULT_MAX_LENGTH = 100
DEFAULT_NUM_BEAMS = 3

def describe_scene(
    frame: Frame, 
    blip_engine: BLIPEngine,
    max_length: int = DEFAULT_MAX_LENGTH,
    num_beams: int = DEFAULT_NUM_BEAMS
) -> str:
    """
    Generate caption for scene in frame.
    
    Args:
        frame: Frame to describe
        blip_engine: BLIPEngine instance
        max_length: Maximum caption length
        num_beams: Beam search width
    
    Returns:
        Scene caption
    """
    caption = blip_engine.caption_image(
        image=frame.image,
        max_length=max_length,
        num_beams=num_beams
    )
    
    if not caption or not caption.strip():
        raise ValueError(f"Empty scene caption for frame {frame.frame_id}")
    
    # Update frame
    frame.caption = caption
    
    logger.debug(f"Frame {frame.frame_id}: \"{caption[:50]}...\"")
    return caption


def batch_describe_scenes(
    frames: List[Frame], 
    blip_engine: BLIPEngine,
    max_length: int = DEFAULT_MAX_LENGTH,
    num_beams: int = DEFAULT_NUM_BEAMS
) -> List[str]:
    """
    Batch generate captions for multiple frames.
    
    Args:
        frames: List of frames to describe
        blip_engine: BLIPEngine instance
        max_length: Maximum caption length
    
    Returns:
        List of captions
    """
    if not frames:
        return []
    
    captions = blip_engine.batch_caption_image(
        images=[frame.image for frame in frames],
        max_length=max_length,
        num_beams=num_beams
    )
    
    # Validate
    if not captions or any(not c.strip() for c in captions):
        raise ValueError("Empty captions in batch")
    
    # Update frames
    for frame, caption in zip(frames, captions):
        frame.caption = caption
    
    logger.info(f"Batch described {len(frames)} frames")
    return captions