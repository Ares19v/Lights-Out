"""
tracker/voice_listener.py
Local voice command listener using openai-whisper (offline, no API key).
Streams mic audio in rolling windows and parses for target keywords.

Requires:
    pip install openai-whisper sounddevice scipy
    ffmpeg on PATH (https://ffmpeg.org/download.html)
"""
from __future__ import annotations
import threading
import time
import queue
import numpy as np
from typing import Callable, Optional

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

try:
    import whisper as _whisper
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False

from tracker.target_resolver import all_keywords, resolve


SAMPLE_RATE = 16000       # Whisper expects 16 kHz
WINDOW_SECS = 3           # Rolling window to transcribe
OVERLAP_SECS = 0.5        # Overlap to catch words at boundaries
TRIGGER_WORDS = ("lock", "track", "target", "aim")


class VoiceListener:
    """
    Background thread that listens to mic, transcribes with Whisper tiny.en,
    and fires on_command(target_str) when a target keyword is heard.
    """

    def __init__(self, on_command: Callable[[str], None]):
        if not _SD_AVAILABLE:
            raise ImportError("sounddevice not installed. Run: pip install sounddevice")
        if not _WHISPER_AVAILABLE:
            raise ImportError("openai-whisper not installed. Run: pip install openai-whisper")

        self.on_command = on_command
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._audio_queue: queue.Queue = queue.Queue()
        self._model = None
        self._last_transcript = ""
        self.last_heard = ""   # For UI display

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    # -- internals ----------------------------------------------------------
    def _run(self):
        print("[Voice] Loading Whisper tiny.en model...")
        try:
            self._model = _whisper.load_model("tiny.en")
            print("[Voice] Whisper ready. Listening for voice commands...")
        except Exception as e:
            print(f"[Voice] Failed to load Whisper: {e}")
            self._running = False
            return

        window_samples = int(SAMPLE_RATE * WINDOW_SECS)
        overlap_samples = int(SAMPLE_RATE * OVERLAP_SECS)
        buffer = np.zeros(0, dtype=np.float32)

        def audio_callback(indata, frames, time_info, status):
            mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
            self._audio_queue.put(mono.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            callback=audio_callback, blocksize=1024):
            while self._running:
                while not self._audio_queue.empty():
                    chunk = self._audio_queue.get_nowait()
                    buffer = np.concatenate([buffer, chunk])

                if len(buffer) >= window_samples:
                    window = buffer[:window_samples]
                    buffer = buffer[window_samples - overlap_samples:]
                    self._transcribe(window)
                else:
                    time.sleep(0.05)

    def _transcribe(self, audio: np.ndarray):
        try:
            result = self._model.transcribe(audio, fp16=False, language="en")
            text = result.get("text", "").strip().lower()
            if not text or text == self._last_transcript:
                return
            self._last_transcript = text
            self.last_heard = text

            print(f"  [Voice] Heard: \"{text}\"")

            for keyword in all_keywords():
                if keyword in text:
                    has_trigger = any(tw in text for tw in TRIGGER_WORDS)
                    is_multiword = len(keyword.split()) > 1
                    if has_trigger or is_multiword:
                        print(f"  [Voice] Command recognized: lock -> '{keyword}'")
                        self.on_command(keyword)
                        return

        except Exception as e:
            print(f"  [Voice] Transcription error: {e}")
