"""Data loader for drone footage - supports video files and frame directories."""

import cv2
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union
import numpy as np
from datetime import datetime, timedelta

from src.storage import FrameStore
from src.models import Frame

class DroneFootageLoader:
    """
    Loads drone footage from video files or frame directories.
    Provides uniform interface for both sources.
    """
    
    def __init__(
        self,
        source: Union[str, Path],
        frames: FrameStore,
        fps: Optional[float] = None,
        record_start_time: Optional[datetime] = None
    ):
        """
        Args:
            source: Path to video file or frames directory
            fps: Override FPS (if None, use video FPS or default 30)
            record_start_time: Recording start time (if None, uses current time)
        """
        self.source = Path(source)
        self.fps = fps
        self.record_start_time = record_start_time or datetime.now()
        self.frames = frames
        
        # Determine source type
        if self.source.is_file():
            self.source_type = "video"
            self._init_video()
        elif self.source.is_dir():
            self.source_type = "frames"
            self._init_frames()
        else:
            raise ValueError(f"Source not found: {source}")
        
        # Eagerly load all frames into store
        self._load_all_frames()
    
    def _init_video(self):
        """Initialize video capture."""
        self.cap = cv2.VideoCapture(str(self.source))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.source}")
        
        # Get video properties
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.fps is None:
            self.fps = self.video_fps
        
        self.frame_paths = None
    
    def _init_frames(self):
        """Initialize frame directory."""
        # Find all image files
        self.frame_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.frame_paths.extend(self.source.glob(ext))
        
        # Sort by frame number (assumes numeric naming like 0.jpg, 1.jpg)
        self.frame_paths = sorted(self.frame_paths, key=lambda x: int(x.stem))
        
        if not self.frame_paths:
            raise ValueError(f"No frames found in: {self.source}")
        
        self.total_frames = len(self.frame_paths)
        
        # Get dimensions from first frame
        first_frame = cv2.imread(str(self.frame_paths[0]))
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {self.frame_paths[0]}")
        
        self.height, self.width = first_frame.shape[:2]
        
        if self.fps is None:
            self.fps = 30.0  # Default FPS for frame sequences
        
        self.cap = None
        self.current_frame_idx = 0
    
    def _load_all_frames(self):
        """Load all frames into the frame store."""
        if self.source_type == "video":
            frame_id = 0
            while True:
                ret, image = self.cap.read()
                if not ret:
                    break
                # Calculate timestamp based on frame_id and FPS
                timestamp_sec = frame_id / self.fps
                captured_at = self.record_start_time + timedelta(seconds=timestamp_sec)
                frame = Frame(
                    frame_id=frame_id,
                    image=image,
                    timestamp_sec=timestamp_sec,
                    captured_at=captured_at,
                )
                self.frames.add(frame)
                frame_id += 1
        else:  # frames directory
            for frame_id, frame_path in enumerate(self.frame_paths):
                image = cv2.imread(str(frame_path))
                if image is not None:
                    # Calculate timestamp based on frame_id and FPS
                    timestamp_sec = frame_id / self.fps
                    captured_at = self.record_start_time + timedelta(seconds=timestamp_sec)
                    frame = Frame(
                        frame_id=frame_id,
                        image=image,
                        timestamp_sec=timestamp_sec,
                        captured_at=captured_at,
                    )
                    self.frames.add(frame)
    
    def __iter__(self) -> Iterator[Frame]:
        """Iterate over frames from the store.
    
        Yields:
            Frame objects
        """
        # Just iterate over the already-loaded frames
        return iter(self.frames)
    
    def __len__(self) -> int:
        """Return total number of frames."""
        return self.total_frames
    
    def info(self) -> dict:
        """Get loader information."""
        return {
            "source": str(self.source),
            "source_type": self.source_type,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.total_frames / self.fps
        }
    
    def close(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()

    def __repr__(self):
        """String representation."""
        info = self.info()
        return (f"DroneFootageLoader(source={info['source_type']}, "
                f"frames={info['total_frames']}, "
                f"fps={info['fps']:.1f}, "
                f"resolution={info['width']}x{info['height']})")