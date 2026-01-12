import time
from typing import Iterator, Optional, Callable
from pathlib import Path
from tqdm import tqdm

from src.models import Frame
from src.storage import FrameStore, ObjectStore
from src.detection import RTDETRDetector, VisualObjectTracker
from src.utils import get_logger

logger = get_logger(__name__)


class Perciever:
    """
    Object detection and tracking.
    
    Features:
    - Periodic detection (every N frames)
    - Continuous tracking (all frames)
    - Automatic sync between detector and tracker
    
    Usage:
        stage = Perciever(loader, detector, tracker, detect_interval=5)
        for frame in stage.process():
            # frame has detected/tracked objects
            print(f"Frame {frame.frame_id}: {len(frame.objects)} objects")
    """
    
    def __init__(
        self,
        frames: FrameStore,
        detector: RTDETRDetector,
        tracker: VisualObjectTracker,
        object_store: ObjectStore,
        detect_interval: int = 5
    ):
        """
        Initialize perciever.
        
        Args:
            frames: Frame store
            detector: Object detector (RT-DETR)
            tracker: Visual tracker (CSRT)
            object_store: Object store for tracked objects
            detect_interval: Run detection every N frames
        """
        self.frames = frames
        self.detector = detector
        self.tracker = tracker
        self.object_store = object_store
        
        self.detect_interval = detect_interval
        
        # State
        self.frame_count = 0
        
        logger.info(
            f"Perciever initialized: "
            f"detect_interval={detect_interval}"
        )
    
    def should_detect(self, frame_id: int) -> bool:
        """Determine if detection should run on this frame."""
        return (frame_id) % self.detect_interval == 0
    
    def process_frame(self, frame: Frame) -> Frame:
        """
        Process a single frame with detection and/or tracking.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with detected/tracked objects
        """
        
        if self.should_detect(frame.frame_id):
            # Run detection
            logger.debug(f"Frame {frame.frame_id}: Running detection")
            frame = self.detector.detect(frame)
            
            # Sync tracker with detections
            frame = self.tracker.sync_with_detections(frame)
        else:
            # Only track existing objects
            logger.debug(f"Frame {frame.frame_id}: Running tracking only")
            frame = self.tracker.update(frame)

        # Store all tracked objects in ObjectStore
        if frame.contains_objects:
            for obj in frame.objects:
                self.object_store.add_object(obj)
        
        self.frame_count += 1
        frame.percieved = True
        
        return frame
    
    def process(
        self,
        on_frame: Optional[Callable[[Frame], None]] = None
    ) -> Iterator[Frame]:
        """
        Process all frames from the loader.
        
        Args:
            on_frame: Optional callback for each processed frame
            show_progress: Whether to show tqdm progress bar
            
        Yields:
            Processed frames with detected/tracked objects
        """
        logger.info("Starting perception pipeline")
        
        # Get total frames if available
        try:
            total_frames = len(self.frames)
        except:
            total_frames = None
        
        # Create progress bar
        pbar = tqdm(
            total=total_frames,
            desc="Processing",
            unit="frame",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
            dynamic_ncols=True,
            ncols=100,
            leave=True,
            position=0
        )
            
        start_time = time.time()
            
        try:
            for frame in self.frames:
                # Process frame
                frame_start = time.time()
                frame = self.process_frame(frame)
                frame_time = time.time() - frame_start
                    
                # Call callback if provided
                if on_frame:
                    on_frame(frame)
                # Update progress bar
                pbar.update(1)
                    
                yield frame
                    
        except Exception as e:
            pbar.close()
            logger.error(f"Error at frame {self.frame_count}: {e}")
            raise
        finally:
            # Close progress bar
            pbar.close()
        
        # Final statistics
        logger.info("Perciever Complete")
    
    def reset(self):
        """Reset stage state."""
        self.tracker = VisualObjectTracker()
        logger.info("Perciever reset")
    
    def __iter__(self) -> Iterator[Frame]:
        """Make the stage directly iterable."""
        return self.process()
