import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from src.models import Frame, Object, BoundingBox
from src.utils import get_logger

logger = get_logger(__name__)

# Color palette for different object classes (BGR format)
COLORS = [
    (128, 0, 0),      # Blue
    (0, 128, 0),      # Green
    (0, 0, 128),      # Red
    (128, 128, 0),    # Cyan
    (128, 0, 128),    # Magenta
    (0, 128, 128),    # Yellow
    (75, 0, 75),      # Purple
    (0, 82, 128),     # Orange
]


def get_color_for_label(label: str) -> Tuple[int, int, int]:
    """
    Get consistent color for a label.
    
    Args:
        label: Object label/class name
        
    Returns:
        BGR color tuple
    """
    # Hash label to get consistent color index
    color_idx = hash(label) % len(COLORS)
    return COLORS[color_idx]


def get_color_for_id(object_id: int) -> Tuple[int, int, int]:
    """
    Get consistent color for a tracking ID.
    
    Args:
        object_id: Object tracking ID
        
    Returns:
        BGR color tuple
    """
    if object_id is None:
        return (128, 128, 128)  # Gray for untracked
    return COLORS[object_id % len(COLORS)]


def draw_bbox(
    image: np.ndarray,
    bbox: BoundingBox,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw a bounding box on the image.
    
    Args:
        image: Input image
        bbox: BoundingBox object
        color: Box color in BGR
        thickness: Line thickness
        
    Returns:
        Image with drawn box
    """
    x1, y1, x2, y2 = [int(v) for v in bbox.to_list()]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image


def draw_label(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    bg_color: Optional[Tuple[int, int, int]] = None,
    font_scale: float = 0.5,
    thickness: int = 1
) -> np.ndarray:
    """
    Draw text label on image with optional background.
    
    Args:
        image: Input image
        text: Text to draw
        position: (x, y) position for text
        color: Text color in BGR
        bg_color: Optional background color
        font_scale: Font size scale
        thickness: Text thickness
        
    Returns:
        Image with drawn text
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = position
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Draw background rectangle if specified
    if bg_color:
        cv2.rectangle(
            image,
            (x, y - text_h - baseline),
            (x + text_w, y + baseline),
            bg_color,
            -1  # Filled
        )
    
    # Draw text
    cv2.putText(
        image, text, (x, y),
        font, font_scale, color, thickness, cv2.LINE_AA
    )
    
    return image


def draw_object(
    image: np.ndarray,
    obj: Object,
    frame_id: int,
    show_id: bool = True,
    show_label: bool = True,
    color_by: str = "id"  # "id" or "label"
) -> np.ndarray:
    """
    Draw a single object on the image.
    
    Args:
        image: Input image
        obj: Object to draw
        frame_id: Current frame ID
        show_id: Whether to show tracking ID
        show_label: Whether to show label
        color_by: Color objects by "id" or "label"
        
    Returns:
        Image with drawn object
    """
    # Get bbox for current frame
    if frame_id not in obj.bounding_boxes:
        return image
    
    bbox = obj.bounding_boxes[frame_id]
    
    # Get color
    if color_by == "id" and obj.object_id is not None:
        color = get_color_for_id(obj.object_id)
    else:
        color = get_color_for_label(obj.label)
    
    # Draw bounding box
    image = draw_bbox(image, bbox, color, thickness=2)
    
    # Prepare label text
    label_parts = []
    if show_id and obj.object_id is not None:
        label_parts.append(f"ID:{obj.object_id}")
    if show_label:
        label_parts.append(obj.label)
    
    if label_parts:
        label_text = " ".join(label_parts)
        x1, y1 = int(bbox.x1), int(bbox.y1)
        
        # Draw label with background
        image = draw_label(
            image, label_text,
            position=(x1, y1 - 5),
            color=(255, 255, 255),  # White text
            bg_color=color,
            font_scale=0.5,
            thickness=1
        )
    
    return image


def visualize_frame(
    frame: Frame,
    show_id: bool = True,
    show_label: bool = True,
    show_info: bool = True,
    color_by: str = "id"
) -> np.ndarray:
    """
    Create visualization of frame with all objects.
    
    Args:
        frame: Frame to visualize
        show_id: Show tracking IDs
        show_label: Show object labels
        show_info: Show frame info overlay
        color_by: Color objects by "id" or "label"
        
    Returns:
        Annotated image
    """
    # Copy image
    vis = frame.image.copy()
    
    # Draw all objects
    for obj in frame.objects:
        vis = draw_object(
            vis, obj, frame.frame_id,
            show_id=show_id,
            show_label=show_label,
            color_by=color_by
        )
    
    # Add frame info overlay
    if show_info:
        info_text = [
            f"Frame: {frame.frame_id}",
            f"Time: {frame.timestamp_sec:.2f}s",
            f"Objects: {len(frame.objects)}"
        ]
        
        y_offset = 30
        for text in info_text:
            vis = draw_label(
                vis, text,
                position=(10, y_offset),
                color=(255, 255, 255),
                bg_color=(0, 0, 0),
                font_scale=0.6,
                thickness=2
            )
            y_offset += 30
    
    return vis


def playback_frames(
    frames: List[Frame],
    fps: float = 30.0,
    window_name: str = "Playback",
    show_id: bool = True,
    show_label: bool = True,
    show_info: bool = True,
    color_by: str = "id"
) -> None:
    """
    Play back a list of frames at specified FPS with interactive controls.
    
    Args:
        frames: List of frames to playback
        fps: Playback frames per second
        window_name: OpenCV window name
        show_id: Show tracking IDs
        show_label: Show object labels
        show_info: Show frame info
        color_by: Color by "id" or "label"
    
    Controls:
        SPACE - Pause/Resume
        i - Toggle IDs
        l - Toggle labels
        c - Toggle color mode (ID/Label)
        h - Toggle info overlay
        q - Quit
    """
    if not frames:
        logger.warning("No frames to playback!")
        return
    
    frame_delay_ms = int(1000 / fps)
    paused = False
    current_show_id = show_id
    current_show_label = show_label
    current_show_info = show_info
    current_color_by = color_by
    
    logger.info(f"Starting playback: {len(frames)} frames at {fps} FPS")
    logger.info("Controls: [SPACE] pause, [i] ID, [l] label, [c] color, [h] info, [q] quit")
    
    try:
        for i, frame in enumerate(frames):
            # Create visualization
            vis = visualize_frame(
                frame,
                show_id=current_show_id,
                show_label=current_show_label,
                show_info=current_show_info,
                color_by=current_color_by
            )
            
            # Add playback indicator
            if paused:
                draw_label(
                    vis, "PAUSED",
                    position=(vis.shape[1] - 150, 30),
                    color=(0, 0, 255),
                    bg_color=(255, 255, 255),
                    font_scale=0.8,
                    thickness=2
                )
            
            # Add playback progress
            progress_text = f"Frame {i+1}/{len(frames)}"
            draw_label(
                vis, progress_text,
                position=(vis.shape[1] - 200, 70),
                color=(255, 255, 255),
                bg_color=(0, 0, 0),
                font_scale=0.6,
                thickness=2
            )
            
            # Show frame
            cv2.imshow(window_name, vis)
            
            # Handle key press
            wait_time = 0 if paused else frame_delay_ms
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q'):
                logger.info("User quit playback")
                break
            elif key == ord(' '):
                paused = not paused
            elif key == ord('i'):
                current_show_id = not current_show_id
            elif key == ord('l'):
                current_show_label = not current_show_label
            elif key == ord('c'):
                current_color_by = "label" if current_color_by == "id" else "id"
            elif key == ord('h'):
                current_show_info = not current_show_info
            
            # If paused, wait for user input
            while paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord(' '):
                    paused = False
                    break
                elif key == ord('q'):
                    logger.info("User quit while paused")
                    paused = False
                    return
                elif key == ord('i'):
                    current_show_id = not current_show_id
                elif key == ord('l'):
                    current_show_label = not current_show_label
                elif key == ord('c'):
                    current_color_by = "label" if current_color_by == "id" else "id"
                elif key == ord('h'):
                    current_show_info = not current_show_info
                
                # Re-render with new settings
                vis = visualize_frame(
                    frame,
                    show_id=current_show_id,
                    show_label=current_show_label,
                    show_info=current_show_info,
                    color_by=current_color_by
                )
                draw_label(
                    vis, "PAUSED",
                    position=(vis.shape[1] - 150, 30),
                    color=(0, 0, 255),
                    bg_color=(255, 255, 255),
                    font_scale=0.8,
                    thickness=2
                )
                progress_text = f"Frame {i+1}/{len(frames)}"
                draw_label(
                    vis, progress_text,
                    position=(vis.shape[1] - 200, 70),
                    color=(255, 255, 255),
                    bg_color=(0, 0, 0),
                    font_scale=0.6,
                    thickness=2
                )
                cv2.imshow(window_name, vis)
    
    finally:
        cv2.destroyWindow(window_name)
        logger.info("Playback complete")


def show_frame(
    frame: Frame,
    window_name: str = "Frame",
    wait_key: int = 1,
    **kwargs
) -> int:
    """
    Display frame in OpenCV window.
    
    Args:
        frame: Frame to display
        window_name: Window name
        wait_key: Milliseconds to wait (0 = wait forever)
        **kwargs: Additional args for visualize_frame
        
    Returns:
        Key pressed by user
    """
    vis = visualize_frame(frame, **kwargs)
    cv2.imshow(window_name, vis)
    return cv2.waitKey(wait_key)

def show_image(
    image: np.ndarray,
    window_name: str = "Image",
    wait_key: int = 1,
) -> int:
    cv2.imshow(window_name, image)
    return cv2.waitKey(wait_key)


def save_frame_visualization(
    frame: Frame,
    output_path: str,
    **kwargs
) -> None:
    """
    Save frame visualization to file.
    
    Args:
        frame: Frame to save
        output_path: Output file path
        **kwargs: Additional args for visualize_frame
    """
    vis = visualize_frame(frame, **kwargs)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis)
    logger.info(f"Saved visualization to {output_path}")

def save_frame(
    image: np.ndarray,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)
    logger.info(f"Saved frame to {output_path}")


class FrameVisualizer:
    """Interactive frame visualizer with keyboard controls."""
    
    def __init__(self, window_name: str = "Frame Visualizer"):
        """Initialize visualizer."""
        self.window_name = window_name
        self.paused = False
        self.show_id = True
        self.show_label = True
        self.show_info = True
        self.color_by = "id"
        
        logger.info("FrameVisualizer initialized")
        logger.info("Controls: [SPACE] pause, [i] toggle ID, [l] toggle label, [c] change color, [q] quit")
    
    def close(self):
        """Close visualizer window."""
        cv2.destroyWindow(self.window_name)
