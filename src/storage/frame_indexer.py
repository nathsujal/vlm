import faiss
import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING

from src.utils import get_logger

if TYPE_CHECKING:
    from src.vlm import CLIPEngine
    from src.models import Frame

logger = get_logger(__name__)

class FrameIndexer:
    """FAISS index manager for frame similarity search."""

    def __init__(self, clip_engine: 'CLIPEngine'):
        """
        Initialize FAISS indexer.
        
        Args:
            clip_engine: CLIPEngine instance for getting embedding dimension
        """
        self.clip_engine = clip_engine
        self.index = faiss.IndexFlatIP(self.clip_engine.embedding_dim)
        self.frame_id_map = []  # Maps index position -> frame_id

        logger.info(
            f"FrameIndexer initialized | "
            f"Index type: IndexFlatIP | "
            f"Embedding dim: {self.clip_engine.embedding_dim}"
        )

    def add_frame(self, frame_id: int, embedding: np.ndarray) -> None:
        """
        Add single frame to index.
        
        Args:
            frame_id: Frame ID
            embedding: Pre-computed embedding
        """
        embedding_array = embedding.reshape(1, -1).astype('float32')
        
        # Add to FAISS index
        self.index.add(embedding_array)
        self.frame_id_map.append(frame_id)
    
    def batch_add_frames(
        self, 
        frame_ids: List[int], 
        embeddings: np.ndarray
    ) -> None:
        """
        Batch add frames to index.
        
        Args:
            frame_ids: List of Frame IDs
            embeddings: Array of embeddings (shape: [N, embedding_dim])
        """
        # Ensure correct shape
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        embeddings_array = embeddings.astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        self.frame_id_map.extend([frame_id for frame_id in frame_ids])
    
    def find_similar_frames(
        self,
        embedding: np.ndarray,
        frame_id: int,
        top_k: int = 5,
        threshold: float = 0.70
    ) -> List[Tuple[int, float]]:
        """
        Find similar frames using FAISS.
        
        Args:
            embedding: Query embedding
            frame_id: Query frame ID (to exclude from results)
            top_k: Number of results
            threshold: Minimum similarity
        
        Returns:
            List of (frame_id, similarity) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Search FAISS
        query_emb = embedding.reshape(1, -1).astype('float32')
        k = min(top_k + 1, self.index.ntotal)  # +1 to account for self
        similarities, indices = self.index.search(query_emb, k)
        
        # Filter and return
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < len(self.frame_id_map):
                result_frame_id = self.frame_id_map[idx]
                # Exclude query frame itself
                if result_frame_id != frame_id and sim < threshold:
                    results.append((result_frame_id, float(sim)))
        
        return results[:top_k]
    
    def is_novel_frame(
        self,
        embedding: np.ndarray,
        frame_id: int,
        similarity_threshold: float = 0.90
    ) -> bool:
        """
        Check if frame is novel.
        
        Args:
            embedding: Query embedding
            frame_id: Query frame ID
            similarity_threshold: Threshold for novelty
        
        Returns:
            True if novel, False if similar to existing
        """
        similar_frames = self.find_similar_frames(
            embedding,
            frame_id,
            top_k=1,
            threshold=similarity_threshold
        )
        
        return len(similar_frames) == 0
    
    def batch_check_novelty(
        self,
        embeddings: np.ndarray,
        frame_ids: List[int],
        similarity_threshold: float = 0.90
    ) -> List[bool]:        
        if len(embeddings) == 0:
            return []
        
        # Ensure correct shape and contiguous array
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        embeddings_array = np.ascontiguousarray(embeddings.astype('float32'))
        
        results = []
        
        # Process each frame incrementally
        for i, frame_id in enumerate(frame_ids):
            frame_emb = embeddings_array[i:i+1]  # Single frame embedding
            
            # If index is empty, first frame is always novel
            if self.index.ntotal == 0:
                results.append(True)
                logger.info(f"Frame {frame_id}: NOVEL (first frame)")
                
                # Index it immediately so next frame can compare
                self.add_frame(frame_id, frame_emb)
                continue
            
            # Check against all previously indexed frames
            k = min(10, self.index.ntotal)
            similarities, indices = self.index.search(frame_emb, k)
            
            is_novel = True
            best_sim = -1.0
            best_frame = None
            
            # Check matches
            for sim, idx in zip(similarities[0], indices[0]):
                if idx == -1 or idx >= len(self.frame_id_map):
                    continue
                
                matched_frame_id = self.frame_id_map[idx]
                
                if sim > best_sim:
                    best_sim = sim
                    best_frame = matched_frame_id
                
                if sim >= similarity_threshold:
                    is_novel = False
                    logger.info(
                        f"Frame {frame_id}: SIMILAR to Frame {best_frame} "
                        f"(sim={sim:.4f})"
                    )
                    break
            
            if is_novel:
                logger.info(
                    f"Frame {frame_id}: NOVEL | "
                    f"Best match: Frame {best_frame} (sim={best_sim:.4f})"
                )
                # Index novel frames immediately
                self.add_frame(frame_id, frame_emb)
            
            results.append(is_novel)
        
        return results
    
    def get_index_size(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal
    
    def reset_index(self) -> None:
        """Clear FAISS index."""
        self.index.reset()
        self.frame_id_map.clear()
        logger.info("FAISS index reset")
    
    def get_stats(self) -> dict:
        """Get indexer statistics."""
        return {
            "index_size": self.index.ntotal,
            "frame_id_map_size": len(self.frame_id_map),
            "index_type": "IndexFlatIP",
            "embedding_dim": self.clip_engine.embedding_dim
        }