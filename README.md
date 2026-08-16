# 🎯 TargetLock

> Real-time GPU-accelerated auto-targeting system powered by MediaPipe & YOLOv8

## Features

- **Text-to-target**: Type `nose`, `chin`, `elbow`, `knee` etc. — instantly locks on
- **Dual model pipeline**: MediaPipe Face Mesh (478 landmarks) + YOLOv8n-Pose (17 keypoints)
- **GPU acceleration**: CUDA/TensorRT via ultralytics + MediaPipe GPU delegate
- **Kalman filter**: Smooth jitter-free tracking, predicts position during brief occlusion
- **Animated HUD**: Crosshair, status ring, confidence bar, FPS counter
- **Manual lock**: Click anywhere in the video window to lock that exact point
- **Live target switching**: Switch targets at any time via console or in-window typing (T key)

## Supported Targets

### Face (MediaPipe — 478 landmarks)
| Keyword | Part |
|---------|------|
| `nose` | Nose tip |
| `chin` | Chin/jaw |
| `forehead` | Forehead center |
| `left eye` / `right eye` | Eye outer corners |
| `left eyebrow` / `right eyebrow` | Eyebrows |
| `mouth` / `lips` / `upper lip` / `lower lip` | Mouth |
| `left cheek` / `right cheek` | Cheeks |
| `left ear` / `right ear` | Ears |

### Body (YOLOv8 COCO-17)
| Keyword | Part |
|---------|------|
| `left shoulder` / `right shoulder` | Shoulders |
| `left elbow` / `right elbow` / `elbow` | Elbows |
| `left wrist` / `right wrist` / `wrist` | Wrists |
| `left hip` / `right hip` / `hip` | Hips |
| `left knee` / `right knee` / `knee` | Knees |
| `left ankle` / `right ankle` / `ankle` | Ankles |

## Installation

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# For NVIDIA GPU users (maximum FPS):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

```bash
# Default (nose target, camera 0, 1280x720)
python main.py

# Custom target
python main.py --target chin

# Custom camera
python main.py --camera 1

# Lower resolution for older hardware
python main.py --width 640 --height 480
```

## Controls

| Control | Action |
|---------|--------|
| `T` (in video window) | Type a new target keyword |
| `Click` (in video window) | Manually lock that exact pixel point |
| `R` | Reset / clear current lock |
| `Q` or `ESC` | Quit |
| Type in console + Enter | Switch target anytime |

## Architecture

```
Webcam → ThreadedCamera (zero-lag bg thread)
              ↓
    ┌─────────────────────┐
    │  FaceTracker        │  MediaPipe Face Mesh (GPU)
    │  PoseTracker        │  YOLOv8n-Pose (CUDA/CPU)
    └─────────────────────┘
              ↓
    TargetResolver (text → landmark index)
              ↓
    KalmanTracker (smooth x,y prediction)
              ↓
    Renderer (crosshair, HUD, labels)
              ↓
    OpenCV Window (native FPS)
```

## Performance Tips

- **NVIDIA GPU**: Install `torch` with CUDA for max FPS (60–120+ FPS)
- **Lower resolution**: Use `--width 640 --height 480` on slower hardware
- **Face-only**: If you only need face targets, the YOLO model won't be needed
