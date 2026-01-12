import numpy as np
from typing import List

from src.vlm import CLIPEngine
from src.models import Frame
from src.utils import get_logger

logger = get_logger(__name__)


def encode_frame(frame: Frame, clip_engine: CLIPEngine) -> np.ndarray:
    """
    Encode a single frame into CLIP embedding.
    
    Args:
        frame: Frame to encode
        clip_engine: CLIPEngine instance
    
    Returns:
        Embedding array (512-dim for ViT-B/32)
    """
    embedding = clip_engine.encode_frame(frame)
    logger.debug(f"Encoded frame {frame.frame_id}")
    return embedding


def batch_encode_frames(
    frames: List[Frame], 
    clip_engine: CLIPEngine
) -> List[np.ndarray]:
    """
    Batch encode multiple frames into CLIP embeddings.
    
    Args:
        frames: List of frames to encode
        clip_engine: CLIPEngine instance
    
    Returns:
        List of embedding arrays
    """
    if not frames:
        return []
    
    embeddings = clip_engine.batch_encode_frames(frames)
    logger.info(f"Batch encoded {len(frames)} frames")
    return embeddings


def get_embedding(frame_id: int, clip_engine: CLIPEngine) -> np.ndarray:
    """
    Retrieve cached embedding for a frame.
    
    Args:
        frame_id: Frame identifier
        clip_engine: CLIPEngine instance
    
    Returns:
        Cached embedding or None
    """
    embedding = clip_engine.get_embedding(frame_id)
    
    if embedding is not None:
        logger.debug(f"Retrieved cached embedding for frame {frame_id}")
    else:
        logger.debug(f"No cached embedding for frame {frame_id}")
    
    return embedding


def compute_similarity(
    emb1: np.ndarray, 
    emb2: np.ndarray
) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding
        emb2: Second embedding
    
    Returns:
        Similarity score (0-1)
    """
    return CLIPEngine.cosine_similarity(emb1, emb2)
