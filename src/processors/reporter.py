import json
from toon import encode
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Optional

from src.storage import FrameStore, ObjectStore, ObjectsAttributeRegistry
from src.models import Frame, Object
from config.settings import settings
from src.utils import get_logger

logger = get_logger(__name__)

class TimelineEvent(BaseModel):
    object_id: int
    object_class: str
    event: str

class TimelineEntry(BaseModel):
    frame_id: int
    timestamp_sec: float
    captured_at: datetime
    events: List[TimelineEvent] = Field(default_factory=list)

class Reporter:
    """
    Intelligence analyst that synthesizes surveillance data into structured reports.
    
    Aggregates frame-level, object-level, and attribute data into timelines
    and insights for LLM consumption.
    """
    
    def __init__(
        self,
        frame_store: FrameStore,
        object_store: ObjectStore,
        object_attribute_registry: ObjectsAttributeRegistry,
        property_data: Optional[List[Dict]] = None
    ):
        self.frames = frame_store
        self.objects = object_store
        self.object_attribute_registry = object_attribute_registry
        self.property_data = property_data or []

        # Timeline of events
        self.timeline: Dict[int, TimelineEntry] = {}  # frame_id -> TimelineEntry
        # Object reports
        self.objects_report: List[Dict[str, str]] = []
        
        logger.info("Analyst initialized")

    def process(self):
        """Generate intelligence reports."""
        logger.info("Building timeline...")
        self._create_timeline()
        logger.info("Building object reports...")
        self._get_all_objects_report()
        logger.info("Saving timeline...")
        self._save_timeline()
        logger.info("Saving object reports...")
        self._save_object_report()
        logger.info("Saving property reports...")
        self._save_property_report()
        logger.info(f"Timeline created with {len(self.timeline)} entries")

    def _create_timeline(self):
        """Build chronological timeline from object events."""
        all_events = self.object_attribute_registry.get_all_events(toon_encode=False)

        for object_id, events in all_events.items():
            obj = self.objects.get_object(object_id)
            if not obj:
                logger.warning(f"Object {object_id} not found in store")
                continue
                
            for event in events:
                # Structure of event
                #  event: ContextualCaption(
                #    frame_id: int
                #    timestamp_sec: float
                #    caption: str
                # )
                timeline_event = TimelineEvent(
                    object_id=object_id,
                    object_class=obj.label,
                    event=event.caption
                )
                
                # Add to or update timeline entry
                if event.frame_id in self.timeline:
                    self.timeline[event.frame_id].events.append(timeline_event)
                else:
                    frame = self.frames.get(event.frame_id)
                    self.timeline[event.frame_id] = TimelineEntry(
                        frame_id=event.frame_id,
                        timestamp_sec=event.timestamp_sec,
                        captured_at=frame.captured_at if frame else datetime.now(),
                        events=[timeline_event]
                    )

    def _save_timeline(self):
        """Save timeline to file."""
        timeline_list = [
            entry.model_dump(mode='json')
            for entry in sorted(self.timeline.values(), key=lambda x: x.frame_id)
        ]
        
        with open(settings.timeline_path, "w") as f:
            json.dump(timeline_list, f, indent=2)

    def _get_all_objects_report(self):
        """Get all objects from store."""
        objects = self.objects.get_all_objects()
        for obj in objects:
            attributes: Dict[str, str] = self.object_attribute_registry.get_attributes(obj.object_id)
            events: List[ContextualCaption] = self.object_attribute_registry.get_events(obj.object_id)
            events_timeline = [
                {
                    "frame_id": e.frame_id,
                    "timestamp_sec": e.timestamp_sec,
                    "captured_at": self.frames.get(e.frame_id).captured_at.isoformat(),  # Convert datetime to string
                    "caption": e.caption
                }
                for e in events
            ]
            self.objects_report.append({
                "object_id": obj.object_id,
                "label": obj.label,
                "duration_sec": round(obj.last_timestamp_sec - (events[0].timestamp_sec if events else 0), 2) if events else 0,
                "attributes": attributes,
                "events": events_timeline,
                "summary_text": self._generate_object_narrative(obj, attributes, events_timeline)
            })
    
    def _generate_object_narrative(self, obj: Object, attributes: Dict, events: List[Dict]) -> str:
        """Generate natural language summary for LLM."""
        # Build attribute description
        attr_desc = f"{attributes.get('color', 'unknown color')} {attributes.get('vehicle_type', 'object')}"
        if 'size_category' in attributes:
            attr_desc += f", {attributes['size_category']} size"
        
        # Calculate duration
        duration_sec = round(obj.last_timestamp_sec - (events[0]['timestamp_sec'] if events else 0), 1) if events else 0
        
        # Build timeline narrative
        if events:
            first_event = events[0]['caption']
            last_event = events[-1]['caption']
            
            narrative = (
                f"Object {obj.object_id} is a {attr_desc}. "
                f"It was tracked for {duration_sec} seconds. "
            )
            
            if len(events) == 1:
                narrative += f"Activity: {first_event}"
            else:
                narrative += f"Initially: {first_event}. Later: {last_event}"
            
            return narrative
        
        return f"Object {obj.object_id} is a {attr_desc}, tracked for {duration_sec} seconds."

    def _save_object_report(self):
        """Save object report to file."""
        with open(settings.object_report_path, "w") as f:
            json.dump(self.objects_report, f, indent=2)
    
    def _save_property_report(self):
        """Save property/scene report to file (novel frames only)."""
        with open(settings.property_report_path, "w") as f:
            json.dump(self.property_data, f, indent=2)
        
        logger.info(f"Saved {len(self.property_data)} novel frame reports to {settings.property_report_path}")