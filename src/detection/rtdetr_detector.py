"""RT-DETR object detector wrapper."""

import cv2
import numpy as np
from typing import List
from pathlib import Path

from ultralytics import RTDETR
import torch

from src.utils import get_logger
from src.models import Frame, Object, BoundingBox

logger = get_logger(__name__)

CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "boat"
]

class RTDETRDetector:
    def __init__(
        self,
        model_size: str = 'l',
        conf_threshold: float = 0.7
    ):
        self.conf_threshold = conf_threshold
        
        # Only detect these classes
        self.allowed_classes = set(CLASSES)

        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        model_name = f"rtdetr-{model_size}.pt"
        logger.info(f"Loading RT-DETR-{model_size.upper()} on {self.device}...")
        self.model = RTDETR(model_name).to(self.device)

        self.class_names = self.model.names

    def detect(
        self,
        frame: Frame
    ) -> Frame:

        results = self.model(frame.image, conf=self.conf_threshold, verbose=False)[0]
        detections = []

        if results.boxes is not None:
            for box in results.boxes.cpu().numpy():
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self.class_names[cls_id]
                
                # Filter: only keep allowed classes
                if class_name not in self.allowed_classes:
                    continue

                detections.append(
                    Object(
                        object_id=None,
                        label=class_name,
                        bounding_boxes={frame.frame_id: BoundingBox(
                            x1=float(x1),
                            y1=float(y1),
                            x2=float(x2),
                            y2=float(y2)
                        )}
                    )
                )
        
        frame.objects = detections

        return frame
