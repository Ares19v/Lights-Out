"""
tracker/face_tracker.py
MediaPipe Face Landmarker wrapper (Tasks API — mediapipe >= 0.10).
Detects 478 facial landmarks and returns (x, y) for a requested index.
"""
from __future__ import annotations
from typing import Optional, Tuple
import os
import time
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# Path to the downloaded model bundle (relative to this file's package root)
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "face_landmarker.task"
)


class FaceTracker:
    """Wraps MediaPipe FaceLandmarker (Tasks API) for single-face landmark extraction."""

    def __init__(self, min_confidence: float = 0.5):
        if not _MP_AVAILABLE:
            raise ImportError("mediapipe is not installed. Run: pip install mediapipe")

        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"Face landmarker model not found at: {_MODEL_PATH}\n"
                "Download it with:\n"
                "  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task "
                "-OutFile models/face_landmarker.task"
            )

        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,   # VIDEO mode for real-time streams
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._detection_result = None
        self._frame_ts_ms: int = 0

    def process(self, frame_rgb: np.ndarray):
        """Run face landmarker on an RGB frame. Call before get_landmark()."""
        # Increment timestamp — must be strictly increasing in VIDEO mode
        self._frame_ts_ms += 1
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._detection_result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

    def get_landmark(self, index: int, frame_w: int, frame_h: int,
                     index2: Optional[int] = None) -> Optional[Tuple[int, int]]:
        """
        Return pixel (x, y) for landmark `index`.
        If index2 is given, return the midpoint of index and index2.
        Returns None if no face detected.
        """
        if not self._detection_result or not self._detection_result.face_landmarks:
            return None

        lms = self._detection_result.face_landmarks[0]

        try:
            lm = lms[index]
            x = int(lm.x * frame_w)
            y = int(lm.y * frame_h)

            if index2 is not None:
                lm2 = lms[index2]
                x = (x + int(lm2.x * frame_w)) // 2
                y = (y + int(lm2.y * frame_h)) // 2

            return x, y
        except IndexError:
            return None

    @property
    def has_face(self) -> bool:
        """True if a face was detected in the last processed frame."""
        return bool(
            self._detection_result and self._detection_result.face_landmarks
        )

    def close(self):
        self._landmarker.close()

