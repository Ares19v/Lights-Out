"""
capture/camera.py
Threaded camera capture to eliminate frame-grab latency.
Runs grab in a background thread so the main loop always gets the freshest frame.
"""
import cv2
import threading
import time


class ThreadedCamera:
    """Non-blocking camera using a daemon thread for continuous frame capture."""

    def __init__(self, src: int = 0, width: int = 1280, height: int = 720):
        self.src = src

        # Try DirectShow first (lowest latency on Windows), fall back to MSMF
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"[Camera] CAP_DSHOW failed, trying default backend...")
            self.cap.release()
            self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {src} — is a camera connected?")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag

        # Try to read first frame — retry a few times (camera warmup)
        ret, frame = False, None
        for attempt in range(10):
            ret, frame = self.cap.read()
            if ret:
                break
            import time; time.sleep(0.1)

        if not ret:
            self.cap.release()
            raise RuntimeError(f"Camera source {src} opened but returned no frames — check permissions or try a different source.")

        self.frame = frame
        self.lock = threading.Lock()
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        """Return the latest available frame."""
        with self.lock:
            return True, self.frame.copy()

    def release(self):
        self.running = False
        self._thread.join(timeout=1.0)
        self.cap.release()

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
