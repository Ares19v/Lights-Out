"""
tracker/one_euro.py
1 Euro Filter implementation for X/Y coordinates.
Dynamically adjusts smoothing based on movement speed.
Eliminates jitter when still, eliminates lag when moving.
"""
import math
import time
from typing import Optional, Tuple

class LowPassFilter:
    def __init__(self, alpha):
        self.a = alpha
        self.y = None
        self.s = None

    def __call__(self, value, alpha=None):
        if alpha is not None:
            self.a = alpha
        if self.y is None:
            s = value
        else:
            s = self.a * value + (1.0 - self.a) * self.s
        self.y = value
        self.s = s
        return s

class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filt = LowPassFilter(self._alpha(1.0/30.0, self.min_cutoff))
        self.dx_filt = LowPassFilter(self._alpha(1.0/30.0, self.d_cutoff))
        self.last_time = None

    def _alpha(self, t_e, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / t_e)

    def __call__(self, x, t):
        if self.last_time is None:
            self.last_time = t
            self.x_filt(x)
            return x

        t_e = t - self.last_time
        if t_e <= 0: t_e = 0.0001
        
        # Filtered derivative
        a_d = self._alpha(t_e, self.d_cutoff)
        dx = (x - self.x_filt.y) / t_e
        dx_hat = self.dx_filt(dx, alpha=a_d)

        # Filtered signal
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(t_e, cutoff)
        x_hat = self.x_filt(x, alpha=a)

        self.last_time = t
        return x_hat


class OneEuroTracker:
    """
    Drop-in replacement for KalmanTracker.
    Tracks X and Y using two independent 1 Euro filters.
    """
    def __init__(self):
        # min_cutoff: lower = less jitter when still, but more lag
        # beta: higher = less lag when moving quickly, but more jitter
        self.filter_x = OneEuroFilter(min_cutoff=0.5, beta=0.01)
        self.filter_y = OneEuroFilter(min_cutoff=0.5, beta=0.01)
        
        self.initialized = False
        self._miss_count = 0
        self.MAX_MISS = 30
        
        self._last_x = 0
        self._last_y = 0

    def update(self, x: float, y: float) -> Tuple[int, int]:
        t = time.time()
        self.initialized = True
        self._miss_count = 0
        
        fx = self.filter_x(x, t)
        fy = self.filter_y(y, t)
        
        self._last_x, self._last_y = int(fx), int(fy)
        return self._last_x, self._last_y

    def predict(self) -> Optional[Tuple[int, int]]:
        """1 Euro doesn't extrapolate like Kalman, so we just decay/hold position."""
        if not self.initialized:
            return None
        self._miss_count += 1
        if self._miss_count > self.MAX_MISS:
            return None
        
        # We don't advance the filter state, we just return the last known good position
        return self._last_x, self._last_y

    def reset(self):
        self.filter_x = OneEuroFilter(min_cutoff=0.5, beta=0.01)
        self.filter_y = OneEuroFilter(min_cutoff=0.5, beta=0.01)
        self.initialized = False
        self._miss_count = 0

    @property
    def is_lost(self) -> bool:
        return self._miss_count > self.MAX_MISS

    @property
    def miss_count(self) -> int:
        return self._miss_count
