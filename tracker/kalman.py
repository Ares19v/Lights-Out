"""
tracker/kalman.py
Kalman Filter wrapper for smooth (x, y) coordinate tracking.
Prevents jitter and provides prediction during brief occlusions.
"""
import cv2
import numpy as np
from typing import Optional, Tuple


class KalmanTracker:
    """
    2D Kalman filter for (x, y) position tracking.
    State: [x, y, dx, dy] — position + velocity
    Measurement: [x, y]
    """

    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)  # 4 state vars, 2 measurement vars

        # Transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix (we measure x, y directly)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process noise — how much the model can change per frame
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # Measurement noise — how much we trust the detector
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0

        # Initial error covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0

        self.initialized = False
        self._miss_count = 0
        self.MAX_MISS = 30  # frames before declaring LOST

    def update(self, x: float, y: float) -> Tuple[int, int]:
        """Feed a new measurement. Returns the smoothed (x, y)."""
        meas = np.array([[x], [y]], dtype=np.float32)

        if not self.initialized:
            # Seed the filter with first observation
            self.kf.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.initialized = True

        self.kf.correct(meas)
        self._miss_count = 0
        pred = self.kf.predict()
        return int(pred[0][0]), int(pred[1][0])

    def predict(self) -> Optional[Tuple[int, int]]:
        """Called when the target is not visible — extrapolates position."""
        if not self.initialized:
            return None
        self._miss_count += 1
        if self._miss_count > self.MAX_MISS:
            return None

        # Dampen velocity during prediction (friction) so it doesn't fly off forever
        self.kf.statePost[2] *= 0.8
        self.kf.statePost[3] *= 0.8

        pred = self.kf.predict()
        
        # Clamp to reasonable coordinates to prevent rendering glitches
        px, py = int(pred[0][0]), int(pred[1][0])
        px = max(-5000, min(5000, px))
        py = max(-5000, min(5000, py))
        
        return px, py

    def reset(self):
        """Reset the filter (e.g. when switching targets)."""
        self.initialized = False
        self._miss_count = 0

    @property
    def is_lost(self) -> bool:
        return self._miss_count > self.MAX_MISS

    @property
    def miss_count(self) -> int:
        return self._miss_count
