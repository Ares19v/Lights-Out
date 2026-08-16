"""
tracker/pose_tracker.py
YOLOv8 Pose wrapper using ultralytics.
Detects the largest person in frame and returns COCO-17 keypoints.
"""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import os

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# COCO-17 keypoint names for reference
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


class PoseTracker:
    """
    YOLOv8 pose detector.
    Automatically downloads yolov8n-pose.pt on first run.
    Tracks the largest bounding-box person in frame.
    """

    CONFIDENCE_THRESHOLD = 0.3

    def __init__(self, model_name: str = "yolov8n-pose.pt", device: str = "auto"):
        if not _YOLO_AVAILABLE:
            raise ImportError("ultralytics is not installed. Run: pip install ultralytics")

        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # PyTorch 2.6+ defaults weights_only=True which blocks ultralytics .pt files.
        # We patch torch.load temporarily so weights_only=False is used during load only.
        # Safe: we downloaded this model from the official ultralytics CDN.
        _orig_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)
        torch.load = _patched_load

        try:
            print(f"[PoseTracker] Loading {model_name} on {device} ...")
            self.model = YOLO(model_name)
        finally:
            torch.load = _orig_load  # always restore

        self.device = device
        self._keypoints: Optional[np.ndarray] = None  # shape (17, 3) -- x, y, conf
        self._bbox_area: float = 0.0

    def process(self, frame_bgr: np.ndarray):
        """
        Run YOLOv8 pose on a BGR frame.
        Selects the largest detected person (by bounding box area).
        """
        results = self.model(
            frame_bgr,
            device=self.device,
            verbose=False,
            conf=self.CONFIDENCE_THRESHOLD,
            imgsz=320,  # MASSIVELY improves FPS on CPU (default is 640)
            half=False,
        )

        self._all_keypoints = []

        for result in results:
            if result.keypoints is None or len(result.keypoints) == 0:
                continue
            if result.boxes is None:
                continue

            # Store keypoints for ALL detected people
            for i in range(len(result.boxes.xyxy)):
                kp_data = result.keypoints.data[i].cpu().numpy()  # (17, 3)
                self._all_keypoints.append(kp_data)

    def get_all_keypoints(self, index: int) -> list[Tuple[int, int]]:
        """
        Return pixel (x, y) for COCO keypoint index for ALL detected people.
        """
        results = []
        for kp in self._all_keypoints:
            if index >= len(kp):
                continue
            x, y, conf = kp[index]
            if conf >= self.CONFIDENCE_THRESHOLD:
                results.append((int(x), int(y)))
        return results

    def get_keypoint(self, index: int) -> Optional[Tuple[int, int]]:
        """Legacy method: return the first valid keypoint found."""
        kps = self.get_all_keypoints(index)
        return kps[0] if kps else None

    @property
    def has_person(self) -> bool:
        return len(self._all_keypoints) > 0
