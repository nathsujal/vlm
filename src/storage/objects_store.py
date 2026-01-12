from typing import Dict, List, Optional, Iterator
from collections import defaultdict, Counter
from datetime import datetime

from src.models import Object
from src.utils import get_logger

logger = get_logger(__name__)

class ObjectStore:
    """
    Centralized storage for tracked objects.
    
    Stores objects detected and tracked by Perciever,
    making them available for ObjectAnalyzer.
    """
    
    def __init__(self):
        """Initialize ObjectStore."""
        # Main storage: track_id -> Object
        self._objects: Dict[int, Object] = {}
        
        # Index by frame for quick lookup
        self._frame_index: Dict[int, List[int]] = defaultdict(list)
        
        logger.info("ObjectStore initialized")
    
    def add_object(
        self,
        obj: Object
    ) -> None:
        """
        Add or update an object in the store.
        
        Accumulates bounding boxes across all frames where object is detected/tracked.
        """
        if obj.object_id not in self._objects:
            # New object - store it
            self._objects[obj.object_id] = obj
        else:
            # Existing object - merge new bounding boxes
            existing_obj = self._objects[obj.object_id]
            
            # Add new bounding boxes from this frame
            existing_obj.bounding_boxes.update(obj.bounding_boxes)
            
            # Update timestamps
            if obj.last_timestamp_sec is not None:
                existing_obj.last_timestamp_sec = obj.last_timestamp_sec
    
    def get_object(self, track_id: int) -> Optional[Object]:
        """Get object by track ID."""
        return self._objects.get(track_id)
    
    def get_all_objects(self) -> List[Object]:
        """Get all tracked objects."""
        return list(self._objects.values())
    
    def get_objects_by_category(self, category: str) -> List[Object]:
        """Get all objects of a specific category."""
        return [
            obj for obj in self._objects.values()
            if obj.label == category
        ]
    
    def __len__(self) -> int:
        """Number of unique tracked objects."""
        return len(self._objects)
    
    def __repr__(self) -> str:
        return f"ObjectStore({len(self)} objects)"
