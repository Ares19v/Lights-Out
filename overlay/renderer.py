"""
overlay/renderer.py
Draws the targeting HUD on top of video frames:
 - Animated crosshair at the locked point
 - Status ring (LOCKED / ACQUIRING / LOST)
 - Target label with emoji
 - FPS counter
 - Confidence bar
 - Mini help text
"""
from __future__ import annotations
import cv2
import numpy as np
import math
import time
from typing import Optional, Tuple

# Colour palette (BGR)
COLOR_LOCKED     = (0, 255, 80)    # green
COLOR_ACQUIRING  = (0, 200, 255)   # yellow
COLOR_LOST       = (50, 50, 255)   # red
COLOR_MANUAL     = (255, 180, 0)   # blue-ish (manual lock)
COLOR_TEXT       = (255, 255, 255) # white
COLOR_BG         = (0, 0, 0)       # black (for text backgrounds)
COLOR_DIM        = (120, 120, 120) # grey

FONT = cv2.FONT_HERSHEY_DUPLEX


class Renderer:
    def __init__(self):
        self._start_time = time.time()
        self._frame_times: list[float] = []
        self._fps: float = 0.0

    # ---------------------------------------------------------------
    # FPS tracking
    # ---------------------------------------------------------------
    def tick(self):
        """Call once per frame to update FPS."""
        now = time.time()
        self._frame_times.append(now)
        # Keep only the last 30 timestamps
        cutoff = now - 1.0
        self._frame_times = [t for t in self._frame_times if t > cutoff]
        self._fps = len(self._frame_times)

    # ---------------------------------------------------------------
    # Main draw call
    # ---------------------------------------------------------------
    def draw(
        self,
        frame: np.ndarray,
        target_pos: Optional[Tuple[int, int]],
        target_label: str,
        target_emoji: str,
        status: str,              # 'LOCKED' | 'ACQUIRING' | 'LOST' | 'MANUAL'
        confidence: float = 1.0,  # 0.0 - 1.0
        miss_frames: int = 0,
        help_text: str = "",
    ) -> np.ndarray:
        """Draw the full HUD onto frame (in-place). Returns frame."""

        h, w = frame.shape[:2]
        t = time.time() - self._start_time  # elapsed time for animation

        color = {
            "LOCKED":    COLOR_LOCKED,
            "ACQUIRING": COLOR_ACQUIRING,
            "LOST":      COLOR_LOST,
            "MANUAL":    COLOR_MANUAL,
        }.get(status, COLOR_DIM)

        # --- Crosshair ---
        if target_pos is not None:
            cx, cy = target_pos
            self._draw_crosshair(frame, cx, cy, color, t, status)

        # --- Status ring label ---
        if target_pos is not None:
            cx, cy = target_pos
            label = f"{target_emoji} {target_label}  [{status}]"
            self._draw_text_bg(frame, label, cx + 18, cy - 18, color, scale=0.55)

        # FPS and Confidence have been moved to the React UI outside the video frame

        # --- Lost counter ---
        if status == "LOST" and miss_frames > 0:
            lt = f"Lost: {miss_frames}f"
            cv2.putText(frame, lt, (10, 30), FONT, 0.5, COLOR_LOST, 1, cv2.LINE_AA)

        # --- Help text (bottom) ---
        if help_text:
            self._draw_text_bg(frame, help_text, 10, h - 15, COLOR_DIM, scale=0.45)

        return frame

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------
    def _draw_crosshair(
        self,
        frame: np.ndarray,
        cx: int, cy: int,
        color: Tuple[int, int, int],
        t: float,
        status: str,
    ):
        """Animated crosshair: 4 corner brackets + pulsing circle."""
        size = 24
        gap  = 8
        thickness = 2

        # Corner bracket arms
        # Top-left
        cv2.line(frame, (cx - size, cy - gap), (cx - size, cy - size), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx - gap, cy - size), (cx - size, cy - size), color, thickness, cv2.LINE_AA)
        # Top-right
        cv2.line(frame, (cx + size, cy - gap), (cx + size, cy - size), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx + gap, cy - size), (cx + size, cy - size), color, thickness, cv2.LINE_AA)
        # Bottom-left
        cv2.line(frame, (cx - size, cy + gap), (cx - size, cy + size), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx - gap, cy + size), (cx - size, cy + size), color, thickness, cv2.LINE_AA)
        # Bottom-right
        cv2.line(frame, (cx + size, cy + gap), (cx + size, cy + size), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx + gap, cy + size), (cx + size, cy + size), color, thickness, cv2.LINE_AA)

        # Pulsing center dot
        pulse_r = int(3 + 2 * abs(math.sin(t * 4)))
        cv2.circle(frame, (cx, cy), pulse_r, color, -1, cv2.LINE_AA)

        # Spinning arc for ACQUIRING state
        if status == "ACQUIRING":
            angle = (t * 200) % 360
            cv2.ellipse(frame, (cx, cy), (size + 4, size + 4),
                        angle, 0, 120, color, 2, cv2.LINE_AA)

        # Static circle for LOCKED / MANUAL
        elif status in ("LOCKED", "MANUAL"):
            cv2.circle(frame, (cx, cy), size + 4, color, 1, cv2.LINE_AA)

    def _draw_text_bg(
        self,
        frame: np.ndarray,
        text: str,
        x: int, y: int,
        color: Tuple[int, int, int],
        scale: float = 0.55,
    ):
        """Draw text with a semi-transparent dark background."""
        thickness = 1 if scale < 0.6 else 2
        (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
        pad = 4
        # Dark background rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay,
                      (x - pad, y - th - pad),
                      (x + tw + pad, y + baseline + pad),
                      COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)
