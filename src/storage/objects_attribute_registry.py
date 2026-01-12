from typing import Dict, List
from toon import encode

from src.models import ContextualCaption
from src.utils import get_logger

logger = get_logger(__name__)

class ObjectsAttributeRegistry:
    """
    Central registry for object attributes and events
    """

    def __init__(self):
        self.registry: Dict[int, Dict[str, str]] = {} # object_id -> attributes
        self.events: Dict[int, List[ContextualCaption]] = {} # object_id -> events

    def register_attributes(self, object_id: int, attributes: Dict[str, str]) -> None:
        """Store attributes for an object."""
        self.registry[object_id] = attributes

    def register_events(self, object_id: int, events: List[ContextualCaption]):
        if object_id not in self.events.keys():
            self.events[object_id] = events
        else:
            self.events[object_id].extend(events)

    def get_attributes(self, object_id: int, toon_encode: bool = False) -> str | Dict[str, str]:
        """Get attributes for an object."""
        attributes = self.registry.get(object_id, {})
        if toon_encode:
            return encode(attributes)
        return attributes
    
    def get_events(self, object_id: int, toon_encode: bool = False) -> str | List[ContextualCaption]:
        events = self.events.get(object_id, [])
        if toon_encode:
            # Convert Pydantic models to dicts for proper serialization
            events_dicts = [caption.model_dump() for caption in events]
            return encode(events_dicts)
        return events

    def get_all_attributes(self, toon_encode: bool = False) -> Dict[int, Dict[str, str]]:
        if toon_encode:
            return encode(self.registry)
        return self.registry

    def get_all_events(self, toon_encode: bool = False) -> Dict[int, List[ContextualCaption]]:
        # Convert Pydantic models to dicts for proper serialization
        if toon_encode:
            events_dict = {}
            for object_id, captions in self.events.items():
                events_dict[object_id] = [caption.model_dump() for caption in captions]
            return encode(events_dict)
        return self.events