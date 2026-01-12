import cv2
import numpy as np
from typing import Dict, Tuple

from src.models import Frame, Object, BoundingBox
from src.utils import get_logger

logger = get_logger(__name__)

class VisualObjectTracker:
    """
    Visual object tracker using OpenCV's CSRT algorithm.
    Tracks objects by their pixel appearance rather than motion prediction.
    """
    
    def __init__(self, iou_threshold: float = 0.3):
        """
        Initialize visual object tracker.
       
        Args:
            iou_threshold: Minimum IoU for matching detections to tracks
        """
        self.trackers: Dict[int, any] = {}  # track_id -> cv2.Tracker
        self.track_metadata: Dict[int, dict] = {}  # track_id -> {label, ...}
        self.next_track_id = 1
        self.iou_threshold = iou_threshold
        
        # Track management
        self.current_frame_id = 0
        self.last_detected_frame: Dict[int, int] = {}  # track_id -> last frame detected
        
        # frame threshold
        self.max_frames_to_keep = 10  # Drop tracks after 10 frames without detection
        
        # Track confidence/quality scores
        self.track_quality: Dict[int, float] = {}  # track_id -> quality score (0-1)
        self.min_track_quality = 0.3  # Drop tracks below this quality
        
        # Store reference appearance for comparison
        self.track_templates: Dict[int, np.ndarray] = {}  # track_id -> reference image
        self.appearance_threshold = 0.4  # Minimum similarity to reference
        
        logger.info(f"VisualObjectTracker initialized (IoU threshold: {iou_threshold})")
        
    def sync_with_detections(self, frame: Frame) -> Frame:
        """
        Sync trackers with new detections.
        Matches existing tracks to detections using IOU to preserve IDs and Classes.
        Adds new trackers for unmatched detections.
        
        Args:
            frame: Frame with detected objects
            
        Returns:
            Frame with updated object IDs and temporal info
        """
        self.current_frame_id = frame.frame_id

        # 1. Update existing trackers to get current positions
        current_tracks = {}  # track_id -> bbox (x, y, w, h)
        tracks_to_remove = []
        
        for track_id, tracker in self.trackers.items():
            success, box = tracker.update(frame.image)
            if success:
                current_tracks[track_id] = box
            else:
                tracks_to_remove.append(track_id)
                
        # Remove failed trackers immediately
        for tid in tracks_to_remove:
            self._remove_track(tid)
            
        # 2. Match detections to tracks
        matched_track_ids = set()
        
        for det in frame.objects:
            # Get bbox for current frame from bounding_boxes dict
            if not det.bounding_boxes or frame.frame_id not in det.bounding_boxes:
                continue
            
            bbox = det.bounding_boxes[frame.frame_id]
            det_box = bbox.to_xywh()  # get bbox in (x, y, width, height) format
            best_iou = 0.0
            best_track_id = None
            
            # Find best matching existing track
            for track_id, track_box in current_tracks.items():
                if track_id in matched_track_ids:
                    continue
                    
                iou = self._calculate_iou(det_box, track_box)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id
            
            # Threshold for matching
            if best_iou > self.iou_threshold and best_track_id is not None:
                # MATCH FOUND: Update existing tracker
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame.image, tuple(int(v) for v in det_box))
                self.trackers[best_track_id] = tracker
                
                # Reset quality score on detection
                self.track_quality[best_track_id] = 1.0
                
                # Update reference template
                template = self._crop_box(frame.image, det_box)
                if template is not None:
                    self.track_templates[best_track_id] = template
                
                # Update object with track ID
                det.object_id = best_track_id
                
                # Update temporal info
                det.last_timestamp_sec = frame.timestamp_sec
                
                # Mark this track as recently detected
                self.last_detected_frame[best_track_id] = self.current_frame_id
                matched_track_ids.add(best_track_id)
                
            else:
                # NO MATCH: New object!
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame.image, tuple(int(v) for v in det_box))
                
                new_id = self.next_track_id
                
                self.trackers[new_id] = tracker
                self.track_metadata[new_id] = {
                    'label': det.label
                }
                
                # Initialize quality
                self.track_quality[new_id] = 1.0
                
                # Store reference template
                template = self._crop_box(frame.image, det_box)
                if template is not None:
                    self.track_templates[new_id] = template
                
                # Update object with track ID and temporal info
                det.object_id = new_id
                det.first_timestamp_sec = frame.timestamp_sec
                det.last_timestamp_sec = frame.timestamp_sec
                
                # Mark as detected
                self.last_detected_frame[new_id] = self.current_frame_id
                matched_track_ids.add(new_id)
                
                self.next_track_id += 1  
        
        # Calculate stats
        num_detections = len(frame.objects)
        num_matched = sum(1 for track_id in matched_track_ids if track_id in self.trackers and track_id != self.next_track_id - 1)
        num_new = num_detections - num_matched
        num_unmatched = len(self.trackers) - num_matched - num_new
        
        # Log sync summary
        logger.debug(
            f"Sync: {num_detections} detections → "
            f"{num_matched} matched, {num_new} new, "
            f"{num_unmatched} unmatched | Total: {len(self.trackers)} active tracks"
        )
        return frame

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (xywh format)."""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def update(self, frame: Frame) -> Frame:
        """
        Update all trackers for the current frame.
        Updates frame.objects with tracked objects.
        
        Args:
            frame: Frame to update
            
        Returns:
            Frame with updated tracked objects
        """
        self.current_frame_id = frame.frame_id
        tracked_objects = []
        lost_track_ids = []
        successful_tracks = 0
        failed_tracks = 0
        
        for track_id, tracker in self.trackers.items():
            success, new_bbox = tracker.update(frame.image)

            if not success:
                failed_tracks += 1
                frames_since_detection = self.current_frame_id - self.last_detected_frame.get(track_id, 0)
                
                if frames_since_detection > self.max_frames_to_keep:
                    lost_track_ids.append(track_id)
                continue
            
            # SOLUTION 1: Stricter time-based removal (reduced to 10 frames)
            frames_since_detection = self.current_frame_id - self.last_detected_frame.get(track_id, 0)
            if frames_since_detection > self.max_frames_to_keep:
                lost_track_ids.append(track_id)
                logger.debug(f"Track {track_id} lost: {frames_since_detection} frames without detection")
                continue

            # SOLUTION 2: Quality degradation over time
            # Degrade quality each frame without detection
            if track_id in self.track_quality:
                self.track_quality[track_id] *= 0.9  # 10% decay per frame
                if self.track_quality[track_id] < self.min_track_quality:
                    lost_track_ids.append(track_id)
                    logger.debug(f"Track {track_id} lost: quality too low ({self.track_quality[track_id]:.2f})")
                    continue

            # SOLUTION 3: Appearance-based validation
            if track_id in self.track_templates:
                current_crop = self._crop_box(frame.image, new_bbox)
                if current_crop is not None:
                    similarity = self._compute_similarity(self.track_templates[track_id], current_crop)
                    if similarity < self.appearance_threshold:
                        lost_track_ids.append(track_id)
                        logger.debug(f"Track {track_id} lost: appearance changed (similarity: {similarity:.2f})")
                        continue

            # SOLUTION 4: Strict boundary checks
            if not self._is_box_valid(new_bbox, frame.image.shape):
                failed_tracks += 1
                frames_since_detection = self.current_frame_id - self.last_detected_frame.get(track_id, 0)
                if frames_since_detection > self.max_frames_to_keep:
                    lost_track_ids.append(track_id)
                continue
            
            # SOLUTION 5: Aggressive drift detection
            if track_id in self.track_metadata and 'last_bbox' in self.track_metadata[track_id]:
                last_bbox = self.track_metadata[track_id]['last_bbox']
                last_cx, last_cy = last_bbox[0] + last_bbox[2]/2, last_bbox[1] + last_bbox[3]/2
                curr_cx, curr_cy = new_bbox[0] + new_bbox[2]/2, new_bbox[1] + new_bbox[3]/2
                
                displacement = ((curr_cx - last_cx)**2 + (curr_cy - last_cy)**2)**0.5
                max_displacement = 80  # Reduced from 200 to 80 pixels
                
                if displacement > max_displacement:
                    failed_tracks += 1
                    logger.debug(f"Track {track_id} drifted {displacement:.0f} pixels, marking as failed")
                    frames_since_detection = self.current_frame_id - self.last_detected_frame.get(track_id, 0)
                    if frames_since_detection > self.max_frames_to_keep:
                        lost_track_ids.append(track_id)
                    continue
            
            # Track succeeded - create object
            successful_tracks += 1
            x, y, w, h = [int(v) for v in new_bbox]
            
            # Create Object for this tracked instance
            metadata = self.track_metadata[track_id]
            
            # Update last known bbox for drift detection
            metadata['last_bbox'] = new_bbox
            
            obj = Object(
                object_id=track_id,
                label=metadata['label'],
                bounding_boxes={frame.frame_id: BoundingBox(x1=x, y1=y, x2=x + w, y2=y + h)},
                last_timestamp_sec=frame.timestamp_sec
            )
            tracked_objects.append(obj)
        
        # Remove lost tracks
        for track_id in lost_track_ids:
            self._remove_track(track_id)
        
        # Log update summary
        if lost_track_ids:
            logger.debug(
                f"Tracking: {successful_tracks} active, {failed_tracks} failed, "
                f"{len(lost_track_ids)} lost"
            )
        
        frame.objects = tracked_objects
        return frame

    def _compute_similarity(self, template: np.ndarray, current: np.ndarray) -> float:
        """
        SOLUTION 3: Compute appearance similarity between template and current crop.
        Uses histogram comparison to detect if tracker is following wrong object.
        
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Resize current to match template size
            if template.shape != current.shape:
                current = cv2.resize(current, (template.shape[1], template.shape[0]))
            
            # Convert to HSV for better color comparison
            template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
            current_hsv = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)
            
            # Compute histograms
            hist_template = cv2.calcHist([template_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist_current = cv2.calcHist([current_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            
            # Normalize
            cv2.normalize(hist_template, hist_template, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_current, hist_current, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # Compare using correlation
            similarity = cv2.compareHist(hist_template, hist_current, cv2.HISTCMP_CORREL)
            
            return max(0.0, similarity)  # Ensure non-negative
        except Exception as e:
            logger.warning(f"Appearance comparison failed: {e}")
            return 1.0  # Assume match if comparison fails

    def _crop_box(self, frame, box):
        """Crop bounding box from frame."""
        x, y, w, h = [int(v) for v in box]
        h_img, w_img = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def _remove_track(self, track_id: int):
        """Safely remove a track and all its metadata."""
        if track_id in self.trackers:
            del self.trackers[track_id]
        if track_id in self.track_metadata:
            del self.track_metadata[track_id]
        if track_id in self.last_detected_frame:
            del self.last_detected_frame[track_id]
        if track_id in self.track_quality:
            del self.track_quality[track_id]
        if track_id in self.track_templates:
            del self.track_templates[track_id]
    
    def _is_box_valid(self, box: tuple, frame_shape: tuple, min_visible_ratio: float = 0.85) -> bool:
        """
        SOLUTION 4: Strict boundary and validity checks.
        
        Args:
            box: (x, y, w, h) bounding box
            frame_shape: (height, width) of frame
            min_visible_ratio: minimum ratio of box that must be inside frame (increased to 0.85)
        
        Returns:
            True if box is valid and sufficiently visible
        """
        x, y, w, h = box
        frame_h, frame_w = frame_shape[:2]
        
        # Check box dimensions are positive
        if w <= 0 or h <= 0:
            return False
        
        # Reject boxes that are too small (likely tracking errors)
        min_size = 15  # Increased from 10 to 15 pixels
        if w < min_size or h < min_size:
            return False
        
        # Reject boxes with unrealistic aspect ratios (tracker drift)
        aspect_ratio = w / h
        if aspect_ratio < 0.15 or aspect_ratio > 8.0:  # Stricter than before
            return False
        
        # Calculate box center
        center_x = x + w / 2
        center_y = y + h / 2
        
        # CRITICAL: Reject if center is outside frame (even slightly)
        margin = 20  # Small margin for objects at edge
        if (center_x < margin or center_x > frame_w - margin or 
            center_y < margin or center_y > frame_h - margin):
            return False
        
        # Calculate visible portion of the box
        visible_x1 = max(0, x)
        visible_y1 = max(0, y)
        visible_x2 = min(frame_w, x + w)
        visible_y2 = min(frame_h, y + h)
        
        visible_width = max(0, visible_x2 - visible_x1)
        visible_height = max(0, visible_y2 - visible_y1)
        visible_area = visible_width * visible_height
        
        total_area = w * h
        
        if total_area <= 0:
            return False
        
        visibility_ratio = visible_area / total_area
        
        # Require strict visibility threshold (85% instead of 70%)
        return visibility_ratio >= min_visible_ratio