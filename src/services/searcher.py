import numpy as np
from typing import List, Tuple

from src.storage import FrameIndexer
from src.vlm import CLIPEngine
from src.models import Frame
from src.utils import get_logger
from src.services import embedder

logger = get_logger(__name__)

def find_similar_frames(
    frame: Frame,
    clip_engine: CLIPEngine,
    indexer: FrameIndexer,
    top_k: int = 5,
    threshold: float = 0.90
) -> List[Tuple[int, float]]:
    """
    Find similar frames to a query frame.
    
    Args:
        frame: Query frame
        clip_engine: CLIPEngine for encoding
        indexer: FrameIndexer for search
        top_k: Number of similar frames to return
        threshold: Minimum similarity threshold
    
    Returns:
        List of (frame_id, similarity) tuples
    """
    # Get embedding
    embedding = embedder.encode_frame(frame, clip_engine)
    
    # Search for similar frames
    similar = indexer.find_similar_frames(
        embedding,
        frame.frame_id,
        top_k=top_k,
        threshold=threshold
    )
    
    if similar:
        logger.debug(
            f"Found {len(similar)} similar frames to frame {frame.frame_id}"
        )
    
    return similar


def check_novelty(
    frame: Frame,
    clip_engine: CLIPEngine,
    indexer: FrameIndexer,
    threshold: float = 0.90
) -> bool:
    """
    Check if a frame is novel (not similar to indexed frames).
    
    Args:
        frame: Frame to check
        clip_engine: CLIPEngine for encoding
        indexer: FrameIndexer for search
        threshold: Similarity threshold for novelty
    
    Returns:
        True if frame is novel, False otherwise
    """
    # Get embedding
    embedding = embedder.encode_frame(frame, clip_engine)
    
    # Check novelty
    is_novel = indexer.is_novel_frame(
        embedding,
        frame.frame_id,
        similarity_threshold=threshold
    )
    
    # Update frame attribute
    frame.is_novel = is_novel
    
    logger.debug(
        f"Frame {frame.frame_id} is {'novel' if is_novel else 'similar to existing'}"
    )
    
    return is_novel


def batch_check_novelty(
    frames: List[Frame],
    clip_engine: CLIPEngine,
    indexer: FrameIndexer,
    threshold: float = 0.90
) -> List[bool]:
    """
    Batch check novelty for multiple frames (optimized).
    
    Args:
        frames: List of frames to check
        clip_engine: CLIPEngine for encoding
        indexer: FrameIndexer for search
        threshold: Similarity threshold for novelty
    
    Returns:
        List of boolean values (True = novel, False = similar)
    """
    if not frames:
        return []
    
    # Batch encode
    embeddings = embedder.batch_encode_frames(frames, clip_engine)
    embeddings_array = np.vstack(embeddings)
    
    # Get frame IDs
    frame_ids = [f.frame_id for f in frames]
    
    # Batch check novelty
    novelty_results = indexer.batch_check_novelty(
        embeddings_array,
        frame_ids,
        similarity_threshold=threshold
    )
    
    # Update frame attributes
    for frame, is_novel in zip(frames, novelty_results):
        frame.is_novel = is_novel
    
    novel_count = sum(novelty_results)
    logger.info(
        f"Batch novelty check: {novel_count}/{len(frames)} frames are novel"
    )
    
    return novelty_results


def find_most_similar_frame(
    frame: Frame,
    clip_engine: CLIPEngine,
    indexer: FrameIndexer,
    threshold: float = 0.90
) -> Tuple[int, float]:
    """
    Find the single most similar frame.
    
    Args:
        frame: Query frame
        clip_engine: CLIPEngine for encoding
        indexer: FrameIndexer for search
        threshold: Minimum similarity threshold
    
    Returns:
        Tuple of (frame_id, similarity) or None if no similar frames
    """
    similar_frames = find_similar_frames(
        frame,
        clip_engine,
        indexer,
        top_k=1,
        threshold=threshold
    )
    
    if similar_frames:
        return similar_frames[0]
    
    return None
