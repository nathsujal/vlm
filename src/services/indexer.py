import numpy as np
from typing import List

from src.storage import FrameIndexer
from src.vlm import CLIPEngine
from src.models import Frame
from src.utils import get_logger
from src.services import embedder

logger = get_logger(__name__)

def index_frame(
    frame: Frame,
    clip_engine: CLIPEngine,
    indexer: FrameIndexer
) -> None:
    """
    Encode and index a single frame.
    
    Args:
        frame: Frame to index
        clip_engine: CLIPEngine for encoding
        indexer: FrameIndexer for FAISS storage
    """
    # Get embedding
    embedding = embedder.encode_frame(frame, clip_engine)
    
    # Add to index
    indexer.add_frame(frame, embedding)
    
    logger.debug(f"Indexed frame {frame.frame_id}")


def batch_index_frames(
    frames: List[Frame],
    clip_engine: CLIPEngine,
    indexer: FrameIndexer
) -> None:
    """
    Batch encode and index multiple frames.
    
    Args:
        frames: List of frames to index
        clip_engine: CLIPEngine for encoding
        indexer: FrameIndexer for FAISS storage
    """
    if not frames:
        return
    
    # Batch encode
    embeddings = embedder.batch_encode_frames(frames, clip_engine)
    
    # Convert to array for FAISS
    embeddings_array = np.vstack(embeddings)
    
    # Batch add to index
    indexer.batch_add_frames(frames, embeddings_array)
    
    logger.info(f"Batch indexed {len(frames)} frames")


def remove_from_index(indexer: FrameIndexer) -> None:
    """
    Clear FAISS index.
    
    Args:
        indexer: FrameIndexer instance
    """
    indexer.reset_index()
    logger.info("FAISS index cleared")


def get_index_size(indexer: FrameIndexer) -> int:
    """
    Get number of indexed frames.
    
    Args:
        indexer: FrameIndexer instance
    
    Returns:
        Number of frames in index
    """
    return indexer.get_index_size()


def get_index_stats(indexer: FrameIndexer) -> dict:
    """
    Get indexer statistics.
    
    Args:
        indexer: FrameIndexer instance
    
    Returns:
        Statistics dictionary
    """
    return indexer.get_stats()
