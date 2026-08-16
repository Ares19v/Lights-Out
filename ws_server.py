"""
ws_server.py
Lightweight asyncio WebSocket server that bridges the TargetLock
processing pipeline to the React frontend.

Protocol (JSON messages):
  Backend -> Frontend:
    { image: <base64 jpeg>, status: str, target: str,
      conf: float, miss: int }

  Frontend -> Backend:
    { cmd: "set_target", target: str }
"""
import asyncio
import base64
import json
import threading
import cv2
import websockets
from typing import Optional, Set


class WSServer:
    """
    Runs a WebSocket server on ws://localhost:8765 in a background thread.
    Call `broadcast_frame(frame, meta)` from the main loop to push frames
    to all connected clients.
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._clients: Set[websockets.WebSocketServerProtocol] = set()
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._target_callback = None   # called when frontend sends set_target

    def set_target_callback(self, fn):
        """Register a callback: fn(target_str) called when frontend requests a target change."""
        self._target_callback = fn

    # ── async internals ──────────────────────────────────────────────────────
    async def _handler(self, ws):
        async with self._lock:
            self._clients.add(ws)
        print(f"[WS] Client connected: {ws.remote_address} | total={len(self._clients)}")
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    if msg.get("cmd") == "set_target" and self._target_callback:
                        target = msg.get("target", "").strip().lower()
                        if target:
                            # Call on the main thread via thread-safe bridge
                            threading.Thread(
                                target=self._target_callback,
                                args=(target,),
                                daemon=True,
                            ).start()
                except json.JSONDecodeError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            async with self._lock:
                self._clients.discard(ws)
            print(f"[WS] Client disconnected | remaining={len(self._clients)}")

    async def _serve(self):
        async with websockets.serve(self._handler, self.host, self.port):
            print(f"[WS] Server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    # ── public API ───────────────────────────────────────────────────────────
    def start(self):
        """Start the WebSocket server in a background daemon thread."""
        t = threading.Thread(target=self._run_loop, daemon=True, name="ws-server")
        t.start()

    def broadcast_frame(
        self,
        frame_bgr,
        status: str = "ACQUIRING",
        target_label: str = "",
        confidence: float = 0.0,
        miss: int = 0,
        jpeg_quality: int = 80,
    ):
        """
        Encode `frame_bgr` as JPEG, base64 it, and broadcast to all WS clients.
        Safe to call from any thread.
        """
        if not self._clients or self._loop is None:
            return

        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        if not ok:
            return

        b64 = base64.b64encode(buf).decode("utf-8")
        payload = json.dumps({
            "image":  b64,
            "status": status,
            "target": target_label,
            "conf":   round(confidence, 3),
            "miss":   miss,
        })

        # Schedule the async broadcast on the server"s own event loop
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    async def _broadcast(self, payload: str):
        dead = set()
        for ws in list(self._clients):
            try:
                await ws.send(payload)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._lock:
                self._clients -= dead
