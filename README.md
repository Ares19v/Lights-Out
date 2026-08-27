<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Share+Tech+Mono&weight=900&size=40&pause=1000&color=FF0000&center=true&vCenter=true&width=600&height=80&lines=LIGHTS-OUT;AUTO+AIMING+SYSTEM;TARGET+ACQUIRED." alt="Lights-Out" />

<br/>

![GitHub stars](https://img.shields.io/github/stars/Ares19v/Lights-Out?style=for-the-badge&color=FF0000&logo=github)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11+cu128-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

<br/>

> **Real-time GPU-accelerated auto-targeting system.**  
> Powered by MediaPipe · YOLOv8 · MiDaS · Whisper · CUDA FP16.

<img src="https://img.shields.io/badge/%E2%9A%A1%20Engine-100%2B%20FPS-ff0000?style=for-the-badge" />
<img src="https://img.shields.io/badge/%F0%9F%8E%AF%20Targets-30%2B%20Landmarks-blueviolet?style=for-the-badge" />
<img src="https://img.shields.io/badge/%F0%9F%8E%99%EF%B8%8F%20Voice-Whisper%20base.en-green?style=for-the-badge" />

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Architecture](#-architecture)
- [Performance](#-performance)
- [Supported Targets](#-supported-targets)
- [Installation](#-installation)
- [Usage](#-usage)
- [Voice Commands](#-voice-commands)
- [Settings & HUD](#️-settings--hud)
- [Tech Stack](#-tech-stack)
- [Roadmap](#-roadmap)

---

## 🧠 Overview

**Lights-Out** is a military-grade autonomous targeting computer that runs entirely on your local machine. It fuses three AI models in parallel to identify, classify, and lock onto any anatomical landmark on a human body in real time.

No cloud. No latency. No mercy.

The system exposes a sleek React dashboard via WebSocket that lets you monitor your feed, configure every parameter on the fly, and control the engine with your voice.

---

## 🎬 Demo

<div align="center">

| WebSocket Dashboard | Voice Command Lock | Multi-Target Mode |
|---|---|---|
| React + Vite live HUD | lock nose → instant lock | Dim secondary targets |

</div>

---

## ⚡ Features

<table>
<tr>
<td width="50%">

### 🎯 Targeting Engine
- **Dual-model inference** — MediaPipe Face Mesh (478 landmarks) runs in parallel with YOLOv8n-Pose (17 body keypoints)
- **30+ named targets** — From 
ose tip to left ankle
- **Kalman Filter prediction** — Continues tracking through brief occlusions
- **Optical Flow tracking** — Lucas-Kanade algorithm bridges AI frames at 60+ FPS
- **Click-to-lock** — Manually lock any pixel on screen

</td>
<td width="50%">

### 🖥️ GPU Pipeline
- **CUDA FP16 inference** — Automatic FP16 half-precision when RTX GPU detected
- **Parallel inference thread** — AI runs in a background thread; video never blocks
- **MiDaS Depth Sorting** — Locks onto the physically *closest* target using Z-axis monocular depth
- **100+ FPS** — Decoupled rendering from AI inference to maximize throughput

</td>
</tr>
<tr>
<td width="50%">

### 🎙️ Voice Control
- **OpenAI Whisper ase.en** — Fully local, offline, no API key
- **Strict syntax** — Must say lock <target> or 	rack <target> — accidental triggers impossible
- **Prompt-biased decoder** — Whisper pre-seeded with all valid target phrases for max accuracy
- **CPU offloaded** — Runs entirely on CPU to keep GPU free for the visual pipeline

</td>
<td width="50%">

### 🌐 React Dashboard
- **WebSocket live feed** — 60-100 FPS JPEG stream to browser
- **Settings panel** — Full config sync in real time: HUD color, crosshair style, confidence threshold
- **Multi-Target Overlay** — Ghost crosshairs on all non-primary targets
- **Gesture Authorization** — Only tracks people raising their hand (wrist above shoulder)
- **Dark + Light themes** — One click

</td>
</tr>
</table>

---

## 🏗️ Architecture

`
┌─────────────────────────────────────────────────────────────────────┐
│                        LIGHTS-OUT ENGINE                            │
│                                                                     │
│  Webcam Input ──► ThreadedCamera (zero-lag background capture)      │
│         │                                                           │
│         ├──► [GPU THREAD] ─────────────────────────────────────┐   │
│         │        YOLOv8n-Pose  (CUDA FP16, imgsz=640)          │   │
│         │        MiDaS Depth   (monocular Z estimation)         │   │
│         │        MediaPipe     (478 Face Landmarks)             │   │
│         │                                                       │   │
│         └──► [MAIN THREAD] ◄──────────────────────────────────┘   │
│                  Optical Flow  (Lucas-Kanade 60 FPS)                │
│                  1 Euro Filter (jitter suppression)                 │
│                  Kalman Filter (occlusion prediction)               │
│                  Renderer      (crosshair, HUD, status ring)        │
│                       │                                             │
│         ┌─────────────┴──────────────┐                             │
│         │                            │                             │
│   [OpenCV Window]          [WebSocket Bridge :8765]                │
│                                      │                             │
│         ┌──────────────────────────┐ │                             │
│         │     VOICE THREAD (CPU)   │ │                             │
│         │  Whisper base.en offline │ │                             │
│         │  3s rolling audio window │ │                             │
│         │  "lock/track <target>"   │ │                             │
│         └──────────────────────────┘ │                             │
│                                      ▼                             │
│                        React Dashboard :5173                        │
└─────────────────────────────────────────────────────────────────────┘
`

---

## 📊 Performance

Benchmarked on **NVIDIA GeForce RTX 5060 Laptop GPU** @ CUDA 12.8

| Mode | Resolution | FPS (Engine) | FPS (WebSocket) |
|------|-----------|-------------|----------------|
| Face Mesh only | 1280×720 | 200+ | ~110 |
| YOLOv8 Pose | 1280×720 | 130+ | ~110 |
| Full Pipeline (Face + Pose + Depth) | 1280×720 | 100+ | ~100 |
| CPU Fallback | 640×480 | 15-30 | ~15-30 |

> WebSocket FPS is bottlenecked by cv2.imencode JPEG compression, not the AI engine.

---

## 🎯 Supported Targets

<details>
<summary><b>👁️ Face Targets (MediaPipe 478-point mesh)</b></summary>

| Keyword | Landmark |
|---------|----------|
| 
ose / 
ose tip | Nose tip (#4) |
| 
orehead | Forehead center (#10) |
| chin / jaw | Chin (#152) |
| left eye / 
ight eye | Eye outer corners |
| left eye inner / 
ight eye inner | Eye inner corners |
| left eyebrow / 
ight eyebrow | Eyebrows |
| mouth / lips | Mouth center (averaged) |
| upper lip / lower lip | Lips individual |
| left cheek / 
ight cheek | Cheekbones |
| left ear / 
ight ear | Ears |

</details>

<details>
<summary><b>💪 Body Targets (YOLOv8 COCO-17 keypoints)</b></summary>

| Keyword | Keypoint |
|---------|----------|
| left shoulder / 
ight shoulder | Shoulder joints (#5, #6) |
| left elbow / 
ight elbow / elbow | Elbow joints (#7, #8) |
| left wrist / 
ight wrist / wrist | Wrist joints (#9, #10) |
| left hip / 
ight hip / hip | Hip joints (#11, #12) |
| left knee / 
ight knee / knee | Knee joints (#13, #14) |
| left ankle / 
ight ankle / nkle | Ankle joints (#15, #16) |

</details>

---

## 🛠️ Installation

### Prerequisites

- Python **3.10+**
- **NVIDIA GPU** with CUDA 12.x for full performance (CPU fallback available)
- **Node.js 18+** for the React dashboard
- **FFmpeg** on PATH for voice commands ([download](https://ffmpeg.org/download.html))

---

### Step 1 — Python Backend

`ash
# Clone the repo
git clone https://github.com/Ares19v/Lights-Out.git

[![CI](https://github.com/Ares19v/Lights-Out/actions/workflows/ci.yml/badge.svg)](https://github.com/Ares19v/Lights-Out/actions/workflows/ci.yml)

cd Lights-Out

# Create a virtual environment
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # Linux/macOS

# Install all pinned dependencies
pip install -r requirements.txt
`

#### 🔥 GPU Acceleration (NVIDIA RTX — Highly Recommended)
`ash
# Uninstall the CPU-only torch that came with requirements.txt
pip uninstall torch torchvision torchaudio -y

# Install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
`

> The engine auto-detects your GPU on startup. FP16 half-precision and imgsz=640 are enabled automatically for RTX cards.

---

### Step 2 — React Frontend

`ash
cd frontend
npm install
`

---

## 🚀 Usage

Open **two terminals** from the project root.

**Terminal 1 — Python Engine:**
`ash
python main.py --ws
`

**Terminal 2 — React Dashboard:**
`ash
cd frontend
npm run dev
`

Then open **http://localhost:5173** in your browser.

### CLI Flags

`
python main.py [OPTIONS]

  --ws            Start WebSocket server on :8765 (required for dashboard)
  --camera N      Camera index (default: 0)
  --target TEXT   Initial target keyword (default: nose)
  --width N       Capture width (default: 1280)
  --height N      Capture height (default: 720)
  --nodepth       Disable MiDaS depth estimation
`

---

## 🎙️ Voice Commands

1. Open the dashboard and toggle **Voice Commands** ON in Settings (⚙️).
2. Wait for MIC: 🟢 indicator to appear on the HUD — Whisper is ready.
3. Speak clearly into your microphone using **strict syntax**:

`
"lock nose"           → 🎯 Locks crosshair to nose tip
"track right wrist"   → 🎯 Tracks right wrist
"lock left shoulder"  → 🎯 Locks to left shoulder joint
"track chin"          → 🎯 Tracks chin/jaw
`

> ⚠️ You **must** say either lock or 	rack before the target. Saying just "nose" is intentionally ignored to prevent accidental triggers.

---

## ⚙️ Settings & HUD

Access via the **⚙️** button next to the title bar.

| Setting | Description |
|---------|-------------|
| **HUD Color** | RGB color of the crosshair and overlays |
| **Crosshair Style** | crosshair / dot / circle / 
eticle |
| **Multi-Target Display** | Show ghost crosshairs on background targets |
| **Voice Commands** | Toggle Whisper listener on/off |
| **Gesture Authorization** | Only track when wrist is raised above shoulder |
| **Threat Assessment** | MiDaS depth sort — closest person is primary target |
| **Confidence Threshold** | YOLO minimum detection confidence (0.0 – 1.0) |
| **Theme** | Dark / Light dashboard |

---

## 🧰 Tech Stack

<div align="center">

| Layer | Technology |
|-------|-----------|
| **AI Vision** | MediaPipe Face Mesh · YOLOv8n-Pose · MiDaS DPT |
| **Voice AI** | OpenAI Whisper ase.en (offline) |
| **GPU Runtime** | PyTorch 2.11 · CUDA 12.8 · FP16 |
| **Tracking Math** | Kalman Filter · Lucas-Kanade Optical Flow · 1 Euro Filter |
| **Backend** | Python 3.10 · asyncio WebSockets |
| **Frontend** | React 19 · Vite · Tailwind CSS |
| **Video Capture** | OpenCV ThreadedCamera |
| **Audio Capture** | sounddevice · scipy |

</div>

---

## 🗺️ Roadmap

- [x] GPU-accelerated YOLOv8 + MediaPipe dual pipeline  
- [x] MiDaS monocular depth for closest-target prioritization  
- [x] Kalman + Optical Flow smooth tracking  
- [x] React WebSocket dashboard  
- [x] Voice command system (Whisper ase.en, strict syntax)  
- [x] Multi-target secondary crosshair display  
- [x] Confidence threshold & gesture authorization  
- [ ] RTSP stream input support (IP cameras)  
- [ ] ONNX export for faster CPU inference  
- [ ] DepthAnything V2 upgrade for metric depth  
- [ ] Mouse servo / hardware control output  

---

<div align="center">

**Built with 🔴 by [Ares19v](https://github.com/Ares19v)**

*For educational and research purposes only.*

</div>