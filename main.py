"""
main.py  —  Lights-Out (Phase 2)
Real-time GPU-accelerated auto-targeting system.
Implements Decoupled Inference, ROI Cropping, 1 Euro Filter, Optical Flow,
Voice Commands, Multi-Target Display, and Confidence Threshold.
"""
import os
os.environ["GLOG_minloglevel"] = "3"  # Suppress MediaPipe INFO/WARNING/ERROR logs

import argparse
import threading
import time
import cv2
import numpy as np
from typing import Optional, Tuple

from capture.camera import ThreadedCamera
from tracker.target_resolver import resolve, all_keywords, TrackerType, TargetDef
from tracker.one_euro import OneEuroTracker
from overlay.renderer import Renderer
from ws_server import WSServer

_face_tracker = None
_pose_tracker = None
_depth_estimator = None


class TargetLock:
    WINDOW_NAME = "Lights-Out"

    def __init__(self, camera_src: int = 0, width: int = 1280, height: int = 720,
                 initial_target: str = "nose", enable_ws: bool = False, enable_depth: bool = False):
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU"
        print(f"╔════════════════════════════════════════════╗")
        print(f"  ═  L I G H T S - O U T  —  Auto Targeting System  ═  ")
        print(f"╚════════════════════════════════════════════╝")
        print(f"  [Device] {'GPU: ' + gpu_name if cuda_ok else 'CPU (install CUDA PyTorch for GPU)'}")

        self.enable_depth = enable_depth
        self.enable_gesture = False
        self.enable_voice = False
        self.enable_multi_target = True  # Show secondary crosshairs by default
        self._voice_listener = None

        # State
        self.current_target: Optional[TargetDef] = None
        self.tracker = OneEuroTracker()
        self.renderer = Renderer()
        self.status = "ACQUIRING"
        self.manual_lock: Optional[Tuple[int, int]] = None
        self.typing_mode = False
        self.typed_text = ""
        self._lock = threading.Lock()

        # Threading / Inference State
        self.running = True
        self.ai_raw_pos = None
        self.ai_all_pts = []      # All detected targets for multi-target display
        self.ai_updated = False
        self.tracked_point = None
        self.prev_gray = None
        self.miss_frames = 0
        self.last_known_pos = None

        # WebSocket server
        self.ws: Optional[WSServer] = None
        if enable_ws:
            self.ws = WSServer()
            self.ws.set_target_callback(self.set_target)
            self.ws.set_shutdown_callback(self.trigger_shutdown)
            self.ws.set_config_callback(self.handle_config)
            self.ws.start()

        # Camera
        print(f"[Camera] Opening source {camera_src} at {width}x{height}...")
        self.camera = ThreadedCamera(camera_src, width, height)
        self.W = self.camera.width
        self.H = self.camera.height
        print(f"[Camera] Ready at {self.W}x{self.H}")

        # Trackers
        self._face = None
        self._pose = None
        self._depth = None
        self._init_trackers()

        # Start background inference thread
        self.inf_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.inf_thread.start()

        self.set_target(initial_target)

    def handle_config(self, key: str, value):
        if key == "gesture":
            self.enable_gesture = value
            print(f"  [Config] Gesture Authorization: {'ENABLED' if value else 'DISABLED'}")
        elif key == "depth":
            self.enable_depth = value
            print(f"  [Config] Threat Assessment (Depth): {'ENABLED' if value else 'DISABLED'}")
            if value and self._depth is None:
                print("[Model] Loading MiDaS Depth Estimator (Dynamic)...")
                try:
                    from tracker.depth_estimator import DepthEstimator
                    self._depth = DepthEstimator()
                    print("[Model] MiDaS Depth Estimator ready")
                except Exception as e:
                    print(f"[Model] Depth unavailable: {e}")
                    self.enable_depth = False
        elif key == "crosshairStyle":
            self.renderer.crosshair_style = value
            print(f"  [Config] Crosshair Style: {value.upper()}")
        elif key == "hudColor":
            self.renderer.hud_color = value
            print(f"  [Config] HUD Color: {value.upper()}")
        elif key == "multiTarget":
            self.enable_multi_target = value
            print(f"  [Config] Multi-Target Display: {'ENABLED' if value else 'DISABLED'}")
        elif key == "confidence":
            thresh = float(value)
            if self._pose:
                self._pose.confidence_threshold = thresh
            print(f"  [Config] Confidence Threshold: {thresh:.0%}")
        elif key == "voice":
            if value:
                self._start_voice()
            else:
                self._stop_voice()


    def _start_voice(self):
        if self._voice_listener:
            self._voice_listener.start()
            self.enable_voice = True
            print("  [Config] Voice Commands: ENABLED")

    def _stop_voice(self):
        if self._voice_listener:
            self._voice_listener.stop()
        self.enable_voice = False
        print("  [Config] Voice Commands: DISABLED")

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

        # Pre-initialize Voice Listener on main thread (fixes PyTorch deadlock)
        try:
            from tracker.voice_listener import VoiceListener
            self._voice_listener = VoiceListener(on_command=self.set_target)
        except Exception as e:
            print(f"[Voice] Initialization failed: {e}")

        if self.enable_depth:
            print("[Model] Loading MiDaS Depth Estimator...")
            try:
                from tracker.depth_estimator import DepthEstimator
                self._depth = DepthEstimator()
                print("[Model] MiDaS Depth Estimator ready")
            except Exception as e:
                print(f"[Model] Depth unavailable: {e}")

    def set_target(self, text: str):
        td = resolve(text)
        if td is None:
            return
        with self._lock:
            self.current_target = td
            self.manual_lock = None
            self.tracker.reset()
            self.tracked_point = None
            self.last_known_pos = None
            self.status = "ACQUIRING"
        print(f"  [->] Target set: {td.emoji} {td.name}")

    def _get_all_raw_positions(self, frame_bgr: np.ndarray) -> list[Tuple[int, int]]:
        """Runs heavy AI inference to find ALL instances of the target."""
        td = self.current_target
        if td is None: return []

        if td.tracker == TrackerType.FACE:
            if self._face is None: return []
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self._face.process(rgb)
            return self._face.get_all_landmarks(td.index, self.W, self.H, td.index2)

        elif td.tracker == TrackerType.POSE:
            if self._pose is None: return []
            
            # --- ROI Cropping Optimization for YOLO ---
            crop_size = 320
            last_pos = self.last_known_pos
            
            # If tracking single target, use ROI
            if last_pos is not None:
                cx, cy = last_pos
                x1 = max(0, cx - crop_size//2)
                y1 = max(0, cy - crop_size//2)
                x2 = min(self.W, cx + crop_size//2)
                y2 = min(self.H, cy + crop_size//2)
                
                if (x2 - x1) >= 150 and (y2 - y1) >= 150:
                    roi = frame_bgr[y1:y2, x1:x2]
                    self._pose.process(roi)
                    pts = self._pose.get_all_keypoints(td.index)
                    if pts:
                        return [(pt[0] + x1, pt[1] + y1) for pt in pts]
            
            # Fallback to full frame
            self._pose.process(frame_bgr)
            return self._pose.get_all_keypoints(td.index)
        return []

    def _inference_worker(self):
        """Dedicated background thread for AI so main loop runs at 60fps."""
        while self.running:
            ret, frame = self.camera.read()
            if not ret or self.current_target is None:
                time.sleep(0.01)
                continue
            all_pts = self._get_all_raw_positions(frame)
            
            # --- Gesture Trigger Logic ---
            # If enabled, only track targets who have their hand raised (wrist higher than shoulder)
            # Y-axis goes DOWN, so smaller Y = higher up.
            if getattr(self, "enable_gesture", False) and self._pose is not None:
                # If we were tracking a face, YOLO might not have run this frame yet.
                if self.current_target and self.current_target.tracker == TrackerType.FACE:
                    self._pose.process(frame)
                    
                valid_pts = []
                for pt in all_pts:
                    # Find the nearest person's skeleton to this point
                    # This is an approximation since all_pts might be from face or pose.
                    # For a true system, we'd extract the specific person index.
                    # But for now, we just enforce that AT LEAST ONE person has a hand raised.
                    has_raised_hand = False
                    try:
                        l_wrists = self._pose.get_all_keypoints(9) # left wrist
                        l_shoulders = self._pose.get_all_keypoints(5) # left shoulder
                        r_wrists = self._pose.get_all_keypoints(10) # right wrist
                        r_shoulders = self._pose.get_all_keypoints(6) # right shoulder
                        
                        for w, s in zip(l_wrists, l_shoulders):
                            if w[1] < s[1]: has_raised_hand = True
                        for w, s in zip(r_wrists, r_shoulders):
                            if w[1] < s[1]: has_raised_hand = True
                    except:
                        pass
                        
                    if has_raised_hand:
                        valid_pts.append(pt)
                
                all_pts = valid_pts

            best_pt = None
            
            if len(all_pts) == 1:
                best_pt = all_pts[0]
            elif len(all_pts) > 1:
                # --- Threat Assessment (Multi-Lock logic) ---
                if self._depth is not None:
                    # Pick the one closest to camera (highest Z score)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    best_score = -1
                    for pt in all_pts:
                        z_score = self._depth.get_depth(rgb, pt[0], pt[1])
                        if z_score > best_score:
                            best_score = z_score
                            best_pt = pt
                else:
                    # Fallback if depth disabled: pick largest bounding box / closest to center
                    cx, cy = self.W // 2, self.H // 2
                    best_pt = min(all_pts, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)

            with self._lock:
                self.ai_raw_pos = best_pt
                self.ai_all_pts = [p for p in all_pts if p != best_pt]  # secondary targets
                self.ai_updated = True
                if best_pt is not None:
                    self.last_known_pos = best_pt

            # Yield briefly to not choke the CPU entirely
            time.sleep(0.005)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self._lock:
                self.manual_lock = (x, y)
                self.tracker.reset()
                self.current_target = None
                self.status = "MANUAL"

    def _console_input_thread(self):
        while True:
            try:
                text = input()
                if text.strip().lower() in ("q", "quit", "exit"):
                    self.running = False
                    break
                if text.strip().lower() == "r":
                    with self._lock:
                        self.manual_lock = None
                        self.tracker.reset()
                else:
                    self.set_target(text.strip())
            except:
                break

    def run(self):
        headless = self.ws is not None
        if not headless:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.WINDOW_NAME, self.W, self.H)
            cv2.setMouseCallback(self.WINDOW_NAME, self._mouse_callback)

        threading.Thread(target=self._console_input_thread, daemon=True).start()

        while self.running:
            ret, frame = self.camera.read()
            if not ret: continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.renderer.tick()

            with self._lock:
                td = self.current_target
                manual = self.manual_lock
                status = self.status
                ai_updated = self.ai_updated
                ai_raw_pos = self.ai_raw_pos
                secondary_pts = list(self.ai_all_pts) if self.enable_multi_target else []
                self.ai_updated = False

            target_pos = None
            confidence = 0.0

            if manual is not None:
                target_pos = self.tracker.update(*manual)
                status = "MANUAL"
                confidence = 1.0
                label = "Manual Point"
                emoji = "📌"

            elif td is not None:
                # 1. Update ground truth from AI if available
                if ai_updated and ai_raw_pos is not None:
                    self.tracked_point = np.array([[ai_raw_pos]], dtype=np.float32)
                    self.miss_frames = 0
                
                # 2. Compute Optical Flow from prev frame
                if self.tracked_point is not None and self.prev_gray is not None:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, self.tracked_point, None, 
                        winSize=(21, 21), maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                    )
                    if st[0][0] == 1:
                        self.tracked_point = p1
                        if not ai_updated:
                            self.miss_frames += 1
                        status = "LOCKED"
                    else:
                        self.tracked_point = None
                        status = "LOST"
                
                # 3. Apply 1 Euro Filter smoothing to the current physical point
                if self.tracked_point is not None:
                    tx, ty = self.tracked_point[0][0]
                    target_pos = self.tracker.update(tx, ty)
                    confidence = max(0.0, 1.0 - (self.miss_frames / 60.0))
                else:
                    target_pos = self.tracker.predict()
                    status = "LOST" if target_pos is not None else "ACQUIRING"
                    confidence = 0.0

                with self._lock:
                    self.status = status

                label = td.name
                emoji = td.emoji
            else:
                label, emoji = "None", "🎯"

            self.prev_gray = gray
            help_txt = "[T] type target  [Click] manual lock  [R] reset  [Q] quit"

            if not headless and self.typing_mode:
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, h-50), (w, h), (0,0,0), -1)
                cv2.putText(frame, f"Target: {self.typed_text}_", (10, h-20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,255,80), 1)

            # Get voice last heard for HUD display
            voice_text = ""
            if self._voice_listener and self.enable_voice:
                voice_text = getattr(self._voice_listener, "last_heard", "")

            frame = self.renderer.draw(
                frame=frame, target_pos=target_pos, target_label=label, target_emoji=emoji,
                status=status, confidence=confidence, miss_frames=self.miss_frames,
                help_text=help_txt if not headless else "",
                secondary_pts=secondary_pts, voice_text=voice_text,
            )

            if not headless:
                cv2.imshow(self.WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27): break
                elif key == ord('r'):
                    with self._lock:
                        self.manual_lock = None
                        self.tracker.reset()
                elif key == ord('t'):
                    self.typing_mode = True
                    self.typed_text = ""
                elif self.typing_mode:
                    if key == 13:
                        self.typing_mode = False
                        if self.typed_text.strip(): self.set_target(self.typed_text.strip())
                    elif key == 8: self.typed_text = self.typed_text[:-1]
                    elif 32 <= key <= 126: self.typed_text += chr(key)
            else:
                if self.ws:
                    self.ws.broadcast_frame(frame, status, label, confidence, self.miss_frames)
                time.sleep(0.001)

            # --- Internal Engine FPS Tracker ---
            if not hasattr(self, "_last_fps_print"):
                self._last_fps_print = time.time()
            if time.time() - self._last_fps_print > 2.0:
                print(f"  [Engine] Raw tracking speed: {self.renderer._fps} FPS")
                self._last_fps_print = time.time()

        self.shutdown()

    def trigger_shutdown(self):
        """Called via WebSocket to safely terminate the daemon."""
        self.running = False

    def shutdown(self):
        self.running = False
        print("\n[Lights-Out] Shutting down...")
        self.camera.release()
        if self._face: self._face.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--target", type=str, default="nose")
    parser.add_argument("--ws", action="store_true")
    parser.add_argument("--depth", action="store_true", help="Enable MiDaS Depth Threat Assessment")
    parser.add_argument("--gesture", action="store_true", help="Require targets to have a raised hand to lock on")
    args = parser.parse_args()

    app = TargetLock(args.camera, args.width, args.height, args.target, args.ws, args.depth)
    app.enable_gesture = args.gesture
    app.run()
