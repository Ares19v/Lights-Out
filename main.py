"""
main.py  —  TargetLock

Real-time GPU-accelerated auto-targeting system.
Locks a crosshair onto user-specified body parts via text input.

Usage:
    python main.py
    python main.py --camera 0
    python main.py --target nose
    python main.py --width 1280 --height 720
"""
import argparse
import threading
import time
import cv2
import numpy as np
from typing import Optional, Tuple

from capture.camera import ThreadedCamera
from tracker.target_resolver import resolve, all_keywords, TrackerType, TargetDef
from tracker.kalman import KalmanTracker
from overlay.renderer import Renderer
from ws_server import WSServer

# Lazy imports so missing optional deps don't crash at start
_face_tracker = None
_pose_tracker = None


# ──────────────────────────────────────────────────────────────────────────────
class TargetLock:
    WINDOW_NAME = "TargetLock"

    def __init__(self, camera_src: int = 0, width: int = 1280, height: int = 720,
                 initial_target: str = "nose", enable_ws: bool = False):
        print(f"╔════════════════════════════════════════════╗")
        print(f"  ═  L I G H T S - O U T  —  Auto Targeting System  ═  ")
        print(f"╚════════════════════════════════════════════╝")
        print("Controls:")
        print("  In-window : Press T to type a new target")
        print("  In-window : Click anywhere to manually lock that point")
        print("  In-window : Press R to reset / clear lock")
        print("  In-window : Press Q or ESC to quit")
        print("  In console: Type target name + Enter at any time")
        print(f"\nAvailable targets: {', '.join(all_keywords())}")
        print()

        # State
        self.current_target: Optional[TargetDef] = None
        self.kalman = KalmanTracker()
        self.renderer = Renderer()
        self.status = "ACQUIRING"
        self.manual_lock: Optional[Tuple[int, int]] = None  # manually-clicked point
        self.typing_mode = False
        self.typed_text = ""
        self._lock = threading.Lock()

        # WebSocket server (optional — for frontend)
        self.ws: Optional[WSServer] = None
        if enable_ws:
            self.ws = WSServer()
            self.ws.set_target_callback(self.set_target)
            self.ws.start()
            print("[WS] WebSocket server started on ws://localhost:8765")

        # Camera
        print(f"[Camera] Opening source {camera_src} at {width}x{height}...")
        self.camera = ThreadedCamera(camera_src, width, height)
        self.W = self.camera.width
        self.H = self.camera.height
        print(f"[Camera] Ready at {self.W}x{self.H}")

        # Trackers (loaded lazily on first use)
        self._face = None
        self._pose = None
        self._init_trackers()

        # Set initial target
        self.set_target(initial_target)

    # ------------------------------------------------------------------
    def _init_trackers(self):
        print("[Model] Loading MediaPipe Face Mesh...")
        try:
            from tracker.face_tracker import FaceTracker
            self._face = FaceTracker()
            print("[Model] MediaPipe Face Landmarker ready")
        except Exception as e:
            print(f"[Model] Face Mesh unavailable: {e}")

        print("[Model] Loading YOLOv8n-Pose...")
        try:
            from tracker.pose_tracker import PoseTracker
            self._pose = PoseTracker(model_name="yolov8n-pose.pt", device="auto")
            print("[Model] YOLOv8n-Pose ready")
        except Exception as e:
            print(f"[Model] Pose unavailable: {e}")

    # ------------------------------------------------------------------
    def set_target(self, text: str):
        """Resolve a text keyword to a TargetDef and reset the Kalman filter."""
        td = resolve(text)
        if td is None:
            print(f"  [!] Unknown target: '{text}'. Type one of: {', '.join(all_keywords())}")
            return
        with self._lock:
            self.current_target = td
            self.manual_lock = None
            self.kalman.reset()
            self.status = "ACQUIRING"
        print(f"  [->] Target set: {td.emoji} {td.name}  ({td.tracker.value} #{td.index})")

    # ------------------------------------------------------------------
    def _get_raw_position(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int]]:
        """Run the appropriate tracker and return raw (x, y) or None."""
        td = self.current_target
        if td is None:
            return None

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if td.tracker == TrackerType.FACE:
            if self._face is None:
                return None
            self._face.process(rgb)
            return self._face.get_landmark(td.index, self.W, self.H, td.index2)

        elif td.tracker == TrackerType.POSE:
            if self._pose is None:
                return None
            self._pose.process(frame_bgr)
            return self._pose.get_keypoint(td.index)

        return None

    # ------------------------------------------------------------------
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self._lock:
                self.manual_lock = (x, y)
                self.kalman.reset()
                self.current_target = None
                self.status = "MANUAL"
            print(f"  [Mouse] Manual lock at ({x}, {y})")

    # ------------------------------------------------------------------
    def _console_input_thread(self):
        """Background thread reading console input."""
        while True:
            try:
                text = input()
                if text.strip().lower() in ("q", "quit", "exit"):
                    break
                if text.strip().lower() == "r":
                    with self._lock:
                        self.manual_lock = None
                        self.kalman.reset()
                    print("  [Reset] Lock cleared")
                else:
                    self.set_target(text.strip())
            except (EOFError, KeyboardInterrupt):
                break

    # ------------------------------------------------------------------
    def run(self):
        headless = self.ws is not None  # WS mode = browser is the display

        if not headless:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, self.W, self.H)
            cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)

        # Start console input thread
        cin_thread = threading.Thread(target=self._console_input_thread, daemon=True)
        cin_thread.start()

        if headless:
            print("\n[TargetLock] Running in HEADLESS mode — open http://localhost:5174 in your browser.")
        else:
            print("\n[TargetLock] Running. Focus the video window and use controls above.")
        print("[TargetLock] Type target names in the console at any time.\n")

        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("[Camera] Frame grab failed, retrying...")
                time.sleep(0.01)
                continue

            self.renderer.tick()

            with self._lock:
                td = self.current_target
                manual = self.manual_lock
                status = self.status

            target_pos = None
            confidence = 0.0

            if manual is not None:
                target_pos = self.kalman.update(*manual)
                status = "MANUAL"
                confidence = 1.0
                label = "Manual Point"
                emoji = "📌"

            elif td is not None:
                raw = self._get_raw_position(frame)

                if raw is not None:
                    target_pos = self.kalman.update(*raw)
                    status = "LOCKED"
                    confidence = 1.0
                    with self._lock:
                        self.status = "LOCKED"
                else:
                    predicted = self.kalman.predict()
                    if predicted is not None:
                        target_pos = predicted
                        status = "LOST"
                        confidence = max(0.0, 1.0 - (self.kalman.miss_count / 30.0))
                    else:
                        target_pos = None
                        status = "ACQUIRING"
                        confidence = 0.0
                    with self._lock:
                        self.status = status

                label = td.name if td else "None"
                emoji = td.emoji if td else "🎯"
            else:
                label = "None"
                emoji = "🎯"

            help_txt = "[T] type target  [Click] manual lock  [R] reset  [Q] quit"

            # In-window typing overlay (only in non-headless mode)
            if not headless and self.typing_mode:
                h, w = frame.shape[:2]
                overlay_txt = f"Target: {self.typed_text}_"
                cv2.rectangle(frame, (0, h-50), (w, h), (0,0,0), -1)
                cv2.putText(frame, overlay_txt, (10, h-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,80), 2)
                help_txt = "Type target name and press ENTER"

            frame = self.renderer.draw(
                frame=frame,
                target_pos=target_pos,
                target_label=label,
                target_emoji=emoji,
                status=status,
                confidence=confidence,
                miss_frames=self.kalman.miss_count,
                help_text=help_txt if not headless else "",
            )

            if not headless:
                cv2.imshow(self.WINDOW_NAME, frame)

            # Broadcast to WebSocket frontend (if enabled)
            if self.ws is not None:
                self.ws.broadcast_frame(
                    frame_bgr=frame,
                    status=status,
                    target_label=label,
                    confidence=confidence,
                    miss=self.kalman.miss_count,
                )

            # Key handling (only when showing a window)
            if not headless:
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('q'), 27):  # Q or ESC
                    break

                elif key == ord('r'):
                    with self._lock:
                        self.manual_lock = None
                        self.kalman.reset()
                    print("  [Reset] Lock cleared")

                elif key == ord('t'):
                    self.typing_mode = True
                    self.typed_text = ""

                elif self.typing_mode:
                    if key == 13:  # Enter
                        self.typing_mode = False
                        if self.typed_text.strip():
                            self.set_target(self.typed_text.strip())
                        self.typed_text = ""
                    elif key == 8:  # Backspace
                        self.typed_text = self.typed_text[:-1]
                    elif 32 <= key <= 126:
                        self.typed_text += chr(key)
            else:
                # Headless — just yield the thread briefly
                time.sleep(0.001)

        self.shutdown()

    # ------------------------------------------------------------------
    def shutdown(self):
        print("\n[TargetLock] Shutting down...")
        self.camera.release()
        if self._face:
            self._face.close()
        cv2.destroyAllWindows()
        print("[TargetLock] Goodbye.")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TargetLock — Real-time auto-targeting system"
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera source index (default: 0)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Camera width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="Camera height (default: 720)")
    parser.add_argument("--target", type=str, default="nose",
                        help="Initial target keyword (default: nose)")
    parser.add_argument("--ws", action="store_true",
                        help="Enable WebSocket server on ws://localhost:8765 for the React frontend")
    args = parser.parse_args()

    app = TargetLock(
        camera_src=args.camera,
        width=args.width,
        height=args.height,
        initial_target=args.target,
        enable_ws=args.ws,
    )
    app.run()
