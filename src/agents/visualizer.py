import cv2
import json
from pathlib import Path
from toon import decode
from typing import List, Dict

from src.models import SecurityAlert
from src.tools import cv_tools
from src.storage import FrameStore
from src.tools.reader import read_objects
from src.utils import get_logger
from config.settings import settings

logger = get_logger(__name__)

# Severity color mappings (BGR format for OpenCV)
SEVERITY_COLORS = {
    'CRITICAL': (0, 0, 255),       # Red
    'HIGH': (0, 165, 255),         # Orange
    'MEDIUM': (0, 255, 255),       # Yellow
    'LOW': (0, 255, 0)             # Green
}


class Visualizer:
    """
    Creates visual threat reports with annotated footage.

    Features:
    - Draw bounding boxes around threat objects
    - Color-coded by severity (RED=CRITICAL, ORANGE=HIGH, YELLOW=MEDIUM, GREEN=LOW)
    - Labels with object ID, class, severity
    - Save annotated frames
    """

    def __init__(self, frames: FrameStore):
        """
        Initialize visualizer.
        
        Args:
            frames: FrameStore containing processed frames with objects
        """
        self.frames = frames
        logger.info(f"Visualizer initialized with FrameStore")

    def visualize_threats(
        self,
        alerts: List[SecurityAlert],
        output_dir: str = settings.visualizer_output_dir,
    ) -> Dict[str, str]:
        """
        Create annotated frames showing threats with bounding boxes.
        
        Args:
            alerts: List of security alerts to visualize
            output_dir: Directory to save annotated frames
            
        Returns:
            Dictionary with output paths
        """
        logger.info(f"🎨 Starting threat visualization for {len(alerts)} alerts")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Extract threat object IDs from alerts
        threat_objects = self._extract_threat_objects(alerts)
        logger.info(f"Found {len(threat_objects)} threat objects to visualize")
        
        # Step 2: Load object tracking data
        object_data = self._load_object_data(threat_objects)
        
        # Step 3: Group alerts by severity for priority drawing
        alerts_by_severity = self._group_alerts_by_severity(alerts)
        
        # Step 4: Load frames from store and annotate
        annotated_frames = self._annotate_frames_from_store(
            threat_objects,
            object_data,
            alerts,
            output_path
        )
        
        logger.info(f"✅ Visualization complete: {len(annotated_frames)} frames saved to {output_dir}")
        
        return {
            'output_dir': str(output_path),
            'annotated_frames': annotated_frames,
            'threat_count': len(threat_objects)
        }
    
    def _extract_threat_objects(self, alerts: List[SecurityAlert]) -> List[int]:
        """Extract unique threat object IDs from alerts."""
        threat_ids = set()
        for alert in alerts:
            if not alert.is_false_positive:
                threat_ids.update(alert.affected_objects)
        return sorted(list(threat_ids))
    
    def _load_object_data(self, threat_object_ids: List[int]) -> Dict:
        """
        Load object tracking data for threat objects.
        
        Returns:
            Dictionary mapping object_id -> object data
        """
        # Read all objects from object_report.json
        objects_data = decode(read_objects())
        
        # Filter to only threat objects
        threat_data = {}
        for obj in objects_data:
            if obj['object_id'] in threat_object_ids:
                threat_data[obj['object_id']] = obj
        
        logger.info(f"Loaded data for {len(threat_data)} threat objects")
        return threat_data
    
    def _group_alerts_by_severity(self, alerts: List[SecurityAlert]) -> Dict:
        """Group alerts by severity level."""
        grouped = {'CRITICAL': [], 'HIGH': [], 'MEDIUM': [], 'LOW': []}
        for alert in alerts:
            if not alert.is_false_positive:
                grouped[alert.severity].append(alert)
        return grouped
    
    def _get_alert_for_object(self, object_id: int, alerts: List[SecurityAlert]) -> SecurityAlert:
        """Find the alert associated with an object."""
        for alert in alerts:
            if object_id in alert.affected_objects:
                return alert
        return None
    
    def _annotate_frames_from_store(
        self,
        threat_objects: List[int],
        object_data: Dict,
        alerts: List[SecurityAlert],
        output_path: Path
    ) -> List[str]:
        """
        Load frames from FrameStore and annotate with threat bounding boxes.
        
        Returns:
            List of saved frame paths
        """
        saved_frames = []
        
        # Get all frames from store
        all_frames = self.frames.get_all()
        logger.info(f"Found {len(all_frames)} frames in FrameStore")

        # Collect frame IDs where threats appear
        frames_with_threats = set()
        for obj_id in threat_objects:
            # Find frames where this threat object appears
            for frame in all_frames:
                print(f"Frame has objects: {frame.contains_objects}")
                for obj in frame.objects:
                    print(f"Object id: {obj.object_id}")
                    if obj.object_id == obj_id:
                        frames_with_threats.add(frame.frame_id)
                        break
        
        logger.info(f"Annotating {len(frames_with_threats)} frames containing threats")
        
        # Process each frame with threats
        for frame in all_frames:
            if frame.frame_id not in frames_with_threats:
                continue
            
            # Annotate the frame
            annotated = self._annotate_frame_from_store(
                frame,
                threat_objects,
                alerts
            )
            
            # Save annotated frame
            frame_filename = f"frame_{frame.frame_id:06d}_annotated.jpg"
            frame_path = output_path / frame_filename
            cv_tools.save_frame(annotated, str(frame_path))
            saved_frames.append(str(frame_path))
            
            if len(saved_frames) % 10 == 0:
                logger.info(f"Processed {len(saved_frames)} threat frames...")
        
        logger.info(f"✅ Saved {len(saved_frames)} annotated frames")
        
        return saved_frames
    
    def _annotate_frame_from_store(
        self,
        frame,  # Frame object from FrameStore
        threat_objects: List[int],
        alerts: List[SecurityAlert]
    ):
        """
        Annotate a Frame from FrameStore with threat bounding boxes.
        
        Args:
            frame: Frame object with image and objects
            threat_objects: List of threat object IDs
            alerts: Security alerts
            
        Returns:
            Annotated image (numpy array)
        """
        annotated = frame.image.copy()
        threat_count = 0
        
        # Draw bounding boxes for each threat object in this frame
        for obj in frame.objects:
            if obj.object_id not in threat_objects:
                continue
            
            # Get alert for this object to determine severity
            alert = self._get_alert_for_object(obj.object_id, alerts)
            if not alert:
                continue
            
            # Get severity color
            color = SEVERITY_COLORS.get(alert.severity, (128, 128, 128))
            
            # Draw bounding box
            bbox = obj.bounding_boxes.get(frame.frame_id)
            if bbox:
                # Draw box
                annotated = cv_tools.draw_bbox(
                    annotated,
                    bbox,
                    color=color,
                    thickness=3
                )
                
                # Draw label above box
                label_text = f"[{alert.severity}] ID:{obj.object_id} {obj.label}"
                x1, y1 = int(bbox.x1), int(bbox.y1)
                
                annotated = cv_tools.draw_label(
                    annotated,
                    label_text,
                    position=(x1, max(y1 - 10, 20)),  # Above box, but not off-screen
                    color=(255, 255, 255),
                    bg_color=color,
                    font_scale=0.6,
                    thickness=2
                )
                
                threat_count += 1
        
        # Add frame info overlay
        info_text = f"Frame: {frame.frame_id} | Threats: {threat_count}"
        annotated = cv_tools.draw_label(
            annotated,
            info_text,
            position=(20, annotated.shape[0] - 30),
            color=(255, 255, 255),
            bg_color=(0, 0, 0),
            font_scale=0.6,
            thickness=2
        )
        
        return annotated