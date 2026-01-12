"""Object detection module initialization."""

from .rtdetr_detector import RTDETRDetector
from .visual_tracker import VisualObjectTracker
__all__ = [
    'RTDETRDetector',
    'VisualObjectTracker'
]

