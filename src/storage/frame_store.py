import numpy as np
import yaml
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Iterator

from src.models import Frame

class FrameStore:
    """Centralized frame storage with state tracking."""

    def __init__(
        self,
        persist_to_disk: bool = False,
        cache_dir: Optional[Path] = None,
        clip_engine: Optional['CLIPEngine'] = None
    ):
        """
        Initialize FrameStore.
        
        Args:
            persist_to_disk: Save frames to disk
            cache_dir: Directory for disk cache
            clip_engine: Optional CLIPEngine for embedding/indexing
        """
        self.persist_to_disk = persist_to_disk
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/frame_cache")

        if self.persist_to_disk:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._frames: Dict[int, Frame] = {}
        self.clip_engine = clip_engine

    def add(self, frame: Frame) -> None:
        """
        Add or update frame.
        
        If CLIPEngine is configured, frame will be automatically
        encoded and indexed (embeddings managed by CLIPEngine).
        """
        self._frames[frame.frame_id] = frame
        
        # Encode and index if CLIP engine available
        if self.clip_engine:
            self.clip_engine.encode_and_index(frame)
        
        if self.persist_to_disk:
            self._save_to_disk(frame)
    
    def batch_add(self, frames: List[Frame]) -> None:
        """
        Batch add frames for better performance.
        
        Args:
            frames: List of frames to add
        """
        # Add to internal storage
        for frame in frames:
            self._frames[frame.frame_id] = frame
        
        # Batch encode and index
        if self.clip_engine:
            self.clip_engine.batch_encode_and_index(frames)
        
        # Save to disk if needed
        if self.persist_to_disk:
            for frame in frames:
                self._save_to_disk(frame)
    
    def get(self, frame_id: int) -> Optional[Frame]:
        """Get frame by ID."""
        if frame_id in self._frames:
            return self._frames[frame_id]
        
        if self.persist_to_disk:
            return self._load_from_disk(frame_id)
        
        return None
    
    def update(self, frame: Frame) -> None:
        """Update existing frame (alias for add)."""
        self.add(frame)
    
    def get_all(self) -> List[Frame]:
        """Get all frames in order."""
        return [self._frames[i] for i in sorted(self._frames.keys())]
    
    def filter_by_state(
        self, 
        percieved: Optional[bool] = None, 
        analyzed: Optional[bool] = None
    ) -> List[Frame]:
        """
        Filter frames by processing state.
        
        Args:
            percieved: Filter by perception status
            analyzed: Filter by analysis status
        
        Returns:
            Filtered list of frames
        """
        frames = self.get_all()
        
        if percieved is not None:
            frames = [f for f in frames if f.percieved == percieved]
        if analyzed is not None:
            frames = [f for f in frames if f.analyzed == analyzed]
        
        return frames
    
    def find_similar_frames(
        self,
        frame: Frame,
        top_k: int = 5,
        threshold: float = 0.90
    ) -> List[Frame]:
        """
        Find similar frames using CLIPEngine.
        
        Args:
            frame: Query frame
            top_k: Number of similar frames to return
            threshold: Minimum similarity threshold
        
        Returns:
            List of similar Frame objects
        """
        if not self.clip_engine:
            return []
        
        # Get similar frame IDs from CLIP engine
        similar_ids = self.clip_engine.find_similar_frames(
            frame, 
            top_k=top_k,
            threshold=threshold
        )
        
        # Retrieve full Frame objects
        return [self.get(frame_id) for frame_id, _ in similar_ids if self.get(frame_id)]
    
    def is_novel_frame(self, frame: Frame, threshold: float = 0.90) -> bool:
        """
        Check if frame is novel using CLIPEngine.
        
        Args:
            frame: Frame to check
            threshold: Similarity threshold
        
        Returns:
            True if novel, False otherwise
        """
        if not self.clip_engine:
            return True  # Assume novel if no CLIP engine
        
        return self.clip_engine.is_novel_frame(frame, threshold)
    
    def _save_to_disk(self, frame: Frame):
        """Save frame to disk."""
        frame_file = self.cache_dir / f"frame_{frame.frame_id:06d}.pkl"
        with open(frame_file, 'wb') as f:
            pickle.dump(frame, f)
    
    def _load_from_disk(self, frame_id: int) -> Optional[Frame]:
        """Load frame from disk."""
        frame_file = self.cache_dir / f"frame_{frame_id:06d}.pkl"
        if frame_file.exists():
            with open(frame_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_metadata(self, path: str):
        """Save frame metadata (not images/embeddings) to JSON."""
        metadata = []
        for frame in self.get_all():
            metadata.append({
                'frame_id': frame.frame_id,
                'timestamp_sec': frame.timestamp_sec,
                'percieved': frame.percieved,
                'analyzed': frame.analyzed,
                'num_objects': len(frame.objects),
                'caption': frame.caption
            })
        
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_stats(self) -> dict:
        """Get store statistics."""
        stats = {
            "total_frames": len(self._frames),
            "percieved_frames": len(self.filter_by_state(percieved=True)),
            "analyzed_frames": len(self.filter_by_state(analyzed=True)),
            "has_clip_engine": self.clip_engine is not None
        }
        
        # Add CLIP stats if available
        if self.clip_engine:
            stats["clip_stats"] = self.clip_engine.get_stats()
        
        return stats
    
    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[Frame]:
        """Iterate frames in order (memory efficient)."""
        for frame_id in sorted(self._frames.keys()):
            yield self.get(frame_id)