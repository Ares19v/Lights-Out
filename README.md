# 🎯 Lights-Out: Auto Aiming System

> Real-time GPU-accelerated auto-targeting system powered by MediaPipe, YOLOv8, MiDaS, and Whisper Voice Commands.

## Features

- **Text-to-target**: Type `nose`, `chin`, `left elbow` - instantly locks on.
- **Voice Commands**: Integrated OpenAI Whisper (`base.en`) running on CPU. Say "lock nose" or "track left shoulder" to instantly switch targets hands-free.
- **Multi-Model Pipeline**: MediaPipe Face Mesh (478 landmarks) + YOLOv8n-Pose (17 keypoints) + MiDaS (Z-axis depth estimation for threat assessment).
- **GPU Acceleration**: CUDA FP16 via ultralytics + PyTorch. Reaches 100+ FPS via decoupled asynchronous inference.
- **Optical Flow & 1 Euro Filter**: Smooth jitter-free 60fps+ crosshair tracking via Lucas-Kanade optical flow.
- **React Dashboard**: Modern web interface connected via WebSockets to control settings.
- **Multi-Target Display**: Draws secondary crosshairs on all detected people in the background.

## Installation (Windows / NVIDIA)

### 1. Python Backend
```bash
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# 2. Install pinned dependencies (reproducible build)
pip install -r requirements.txt

# 3. Install PyTorch with CUDA 12.8 support (Required for GPU Acceleration)
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. Voice Commands Requirement
# Make sure you have `ffmpeg` installed and added to your system PATH!
# Download from: https://ffmpeg.org/download.html
```

### 2. React Frontend
```bash
cd frontend
npm install
```

## Usage

You must run both the backend engine and the frontend dashboard.

**Start the Backend (Terminal 1):**
```bash
# From the project root
python main.py --ws
```

**Start the Frontend (Terminal 2):**
```bash
cd frontend
npm run dev
```

Open your browser to `http://localhost:5173` to access the dashboard.

