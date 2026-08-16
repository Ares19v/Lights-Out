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
TRIGGER_WORDS = ("lock", "track")

class VoiceListener:
    """
    Background thread that listens to mic, transcribes with Whisper base.en,
    and strictly fires on_command(target_str) when "lock/track <target>" is heard.
    """

    def __init__(self, on_command: Callable[[str], None]):
        if not _SD_AVAILABLE:
            print("[Voice] WARNING: sounddevice not installed. Voice disabled.")
        if not _WHISPER_AVAILABLE:
            print("[Voice] WARNING: openai-whisper not installed. Voice disabled.")
            
        self.on_command = on_command
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._audio_queue = queue.Queue()
        self._model = None
        self._last_transcript = ""
        self.last_heard = ""

        # Pre-build a strict trigger prompt to bias Whisper to listen for these words
        self._prompt = "Commands: " + ", ".join([f"lock {kw}" for kw in list(all_keywords())[:10]])

        # Pre-load on main thread to avoid Windows PyTorch deadlocks
        if _WHISPER_AVAILABLE:
            print("[Voice] Loading Whisper base.en model on main thread...")
            try:
                self._model = _whisper.load_model("base.en", device="cpu")
                print("[Voice] Whisper ready. Waiting for activation...")
            except Exception as e:
                print(f"[Voice] Failed to load Whisper: {e}")

    def start(self):
        if not _SD_AVAILABLE or not _WHISPER_AVAILABLE or self._model is None:
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    # -- internals ----------------------------------------------------------
    def _run(self):

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
            result = self._model.transcribe(audio, fp16=False, language="en", initial_prompt=self._prompt)
            text = result.get("text", "").strip().lower()
            
            # Print ALL heard text for debugging
            if text:
                print(f"  [Voice Raw] {text}")
                
            if not text or text == self._last_transcript:
                return
            self._last_transcript = text
            self.last_heard = text

            for keyword in all_keywords():
                for trigger in TRIGGER_WORDS:
                    # Strict enforcement: must say "lock nose" or "track nose"
                    if f"{trigger} {keyword}" in text:
                        print(f"  [Voice] Command recognized: {trigger} -> '{keyword}'")
                        self.on_command(keyword)
                        return

        except Exception as e:
            print(f"  [Voice] Transcription error: {e}")
