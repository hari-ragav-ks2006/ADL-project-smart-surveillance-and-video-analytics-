from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """Represents a single detected and tracked object in a frame."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixels

    @property
    def center(self) -> Tuple[int, int]:
        """Centroid of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @property
    def bottom_center(self) -> Tuple[int, int]:
        """Ground contact point (bottom-center) for accurate zone testing."""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int(y2))

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


# Standard COCO class ID mapping
COCO_CLASS_NAMES: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
}


class ObjectDetectorTracker:
    """
    General-purpose real-time object detector and tracker.
    
    Tracking Algorithm Decision:
    - ByteTrack is selected over DeepSORT for CPU/embedded efficiency.
    - ByteTrack associates every detection box (including low-score boxes) using Kalman filtering
      and motion similarity, preventing false fragmentations without requiring an expensive secondary
      Re-ID CNN embedding network per object.
    - Achieves 50-80+ FPS on CPU vs 15-20 FPS with DeepSORT on typical laptop processors.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        tracker_type: str = "bytetrack.yaml",
        target_classes: Optional[List[int]] = None,
        device: str = "auto",
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.tracker_type = tracker_type
        self.target_classes = target_classes or [0]

        # Auto-detect optimal compute device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
                try:
                    torch.set_num_threads(1)
                except Exception:
                    pass
        else:
            self.device = device
            if self.device == "cpu":
                try:
                    torch.set_num_threads(1)
                except Exception:
                    pass

        import threading
        self._lock = threading.Lock()
        logger.info(f"[CORE ENGINE] Initializing YOLOv8 '{model_name}' on device: {self.device}")
        self.model = YOLO(model_name)
        self.model.to(self.device)
        self.next_simulated_id = 100
        logger.info(f"[CORE ENGINE] Tracker: {self.tracker_type} | Target classes: {self.target_classes}")

    def track_frame(
        self,
        frame: np.ndarray,
        persist: bool = True,
    ) -> List[TrackedObject]:
        """
        Run object detection and multi-object tracking on an input frame.
        
        Args:
            frame: Input BGR image (numpy ndarray)
            persist: Keep track state across calls
            
        Returns:
            List of TrackedObject instances present in the current frame.
        """
        if frame is None or frame.size == 0:
            return []

        # Run tracking through ultralytics with thread safety
        with self._lock:
            results = self.model.track(
                source=frame,
                persist=persist,
                tracker=self.tracker_type,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.target_classes,
                device=self.device,
                verbose=False,
            )

        tracked_objects: List[TrackedObject] = []

        if not results or len(results) == 0:
            return tracked_objects

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return tracked_objects

        # Extract coordinates, confidences, class IDs, and tracking IDs
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            # Track ID assigned by ByteTrack (if unassigned, generate fallback)
            if box.id is not None:
                track_id = int(box.id[0].item())
            else:
                track_id = self.next_simulated_id
                self.next_simulated_id += 1

            class_name = self.model.names.get(cls_id, COCO_CLASS_NAMES.get(cls_id, f"obj_{cls_id}"))

            tracked_objects.append(
                TrackedObject(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                )
            )

        # Filter strictly by target_classes
        if self.target_classes is not None:
            tracked_objects = [obj for obj in tracked_objects if obj.class_id in self.target_classes]

        # Deduplicate heavily overlapping boxes of the same class (IoU > 0.45)
        final_objects: List[TrackedObject] = []
        sorted_objects = sorted(tracked_objects, key=lambda o: o.confidence, reverse=True)
        for obj in sorted_objects:
            keep = True
            for accepted in final_objects:
                if obj.class_id == accepted.class_id:
                    xA = max(obj.bbox[0], accepted.bbox[0])
                    yA = max(obj.bbox[1], accepted.bbox[1])
                    xB = min(obj.bbox[2], accepted.bbox[2])
                    yB = min(obj.bbox[3], accepted.bbox[3])
                    inter_w = max(0, xB - xA)
                    inter_h = max(0, yB - yA)
                    inter_area = inter_w * inter_h
                    box1_area = (obj.bbox[2] - obj.bbox[0]) * (obj.bbox[3] - obj.bbox[1])
                    box2_area = (accepted.bbox[2] - accepted.bbox[0]) * (accepted.bbox[3] - accepted.bbox[1])
                    union_area = box1_area + box2_area - inter_area
                    iou = inter_area / float(union_area) if union_area > 0 else 0
                    if iou > 0.45:
                        keep = False
                        break
            if keep:
                final_objects.append(obj)

        return final_objects
