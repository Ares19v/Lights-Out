"""
tracker/target_resolver.py
Maps user-typed text strings to landmark indices for either
MediaPipe Face Mesh or YOLOv8 Pose models.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TrackerType(Enum):
    FACE = "face"   # MediaPipe Face Mesh
    POSE = "pose"   # YOLOv8 Pose


@dataclass
class TargetDef:
    name: str                  # Human-readable label
    tracker: TrackerType       # Which model to query
    index: int                 # Landmark/keypoint index
    # For FACE targets that use two landmarks averaged (e.g. mouth center)
    index2: Optional[int] = None
    emoji: str = "🎯"


# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe Face Mesh landmark indices (478-point model)
# ──────────────────────────────────────────────────────────────────────────────
FACE_TARGETS: dict[str, TargetDef] = {
    # Nose
    "nose":               TargetDef("Nose Tip",         TrackerType.FACE, 4,   emoji="👃"),
    "nose tip":           TargetDef("Nose Tip",         TrackerType.FACE, 4,   emoji="👃"),
    "nose bridge":        TargetDef("Nose Bridge",      TrackerType.FACE, 168, emoji="👃"),
    "nose root":          TargetDef("Nose Root",        TrackerType.FACE, 197, emoji="👃"),
    "left nostril":       TargetDef("L. Nostril",       TrackerType.FACE, 98,  emoji="👃"),
    "right nostril":      TargetDef("R. Nostril",       TrackerType.FACE, 327, emoji="👃"),
    
    # Chin / Jaw
    "chin":               TargetDef("Chin",             TrackerType.FACE, 152, emoji="🫦"),
    "jaw":                TargetDef("Chin",             TrackerType.FACE, 152, emoji="🫦"),
    "left jaw":           TargetDef("L. Jaw Angle",     TrackerType.FACE, 132, emoji="🦴"),
    "right jaw":          TargetDef("R. Jaw Angle",     TrackerType.FACE, 361, emoji="🦴"),
    
    # Forehead
    "forehead":           TargetDef("Forehead",         TrackerType.FACE, 10,  emoji="🧠"),
    "left temple":        TargetDef("L. Temple",        TrackerType.FACE, 162, emoji="🧠"),
    "right temple":       TargetDef("R. Temple",        TrackerType.FACE, 389, emoji="🧠"),
    
    # Eyes
    "left eye":           TargetDef("Left Eye",         TrackerType.FACE, 468, emoji="👁️"),
    "right eye":          TargetDef("Right Eye",        TrackerType.FACE, 473, emoji="👁️"),
    "left pupil":         TargetDef("L. Pupil",         TrackerType.FACE, 468, emoji="👁️"),
    "right pupil":        TargetDef("R. Pupil",         TrackerType.FACE, 473, emoji="👁️"),
    "left eye inner":     TargetDef("L. Eye Inner",     TrackerType.FACE, 133, emoji="👁️"),
    "right eye inner":    TargetDef("R. Eye Inner",     TrackerType.FACE, 362, emoji="👁️"),
    "left eye outer":     TargetDef("L. Eye Outer",     TrackerType.FACE, 33,  emoji="👁️"),
    "right eye outer":    TargetDef("R. Eye Outer",     TrackerType.FACE, 263, emoji="👁️"),
    "left eye top":       TargetDef("L. Eye Top",       TrackerType.FACE, 159, emoji="👁️"),
    "left eye bottom":    TargetDef("L. Eye Bottom",    TrackerType.FACE, 145, emoji="👁️"),
    "right eye top":      TargetDef("R. Eye Top",       TrackerType.FACE, 386, emoji="👁️"),
    "right eye bottom":   TargetDef("R. Eye Bottom",    TrackerType.FACE, 374, emoji="👁️"),
    
    # Eyebrows
    "left eyebrow":       TargetDef("L. Eyebrow",       TrackerType.FACE, 105, emoji="⬇️"),
    "right eyebrow":      TargetDef("R. Eyebrow",       TrackerType.FACE, 334, emoji="⬇️"),
    "left eyebrow inner": TargetDef("L. Brow Inner",    TrackerType.FACE, 55,  emoji="⬇️"),
    "left eyebrow outer": TargetDef("L. Brow Outer",    TrackerType.FACE, 46,  emoji="⬇️"),
    "right eyebrow inner":TargetDef("R. Brow Inner",    TrackerType.FACE, 285, emoji="⬇️"),
    "right eyebrow outer":TargetDef("R. Brow Outer",    TrackerType.FACE, 276, emoji="⬇️"),
    
    # Mouth / Lips
    "mouth":              TargetDef("Mouth Center",     TrackerType.FACE, 13, 14, emoji="👄"),
    "lips":               TargetDef("Lips",             TrackerType.FACE, 13, 14, emoji="👄"),
    "upper lip":          TargetDef("Upper Lip",        TrackerType.FACE, 13,  emoji="👄"),
    "lower lip":          TargetDef("Lower Lip",        TrackerType.FACE, 14,  emoji="👄"),
    "upper lip top":      TargetDef("Upper Lip Edge",   TrackerType.FACE, 0,   emoji="👄"),
    "lower lip bottom":   TargetDef("Lower Lip Edge",   TrackerType.FACE, 17,  emoji="👄"),
    "left mouth corner":  TargetDef("L. Mouth Corner",  TrackerType.FACE, 61,  emoji="👄"),
    "right mouth corner": TargetDef("R. Mouth Corner",  TrackerType.FACE, 291, emoji="👄"),
    
    # Cheeks
    "left cheek":         TargetDef("Left Cheek",       TrackerType.FACE, 50,  emoji="😊"),
    "right cheek":        TargetDef("Right Cheek",      TrackerType.FACE, 280, emoji="😊"),
}

# ──────────────────────────────────────────────────────────────────────────────
# YOLOv8 Pose COCO-17 keypoint indices
# ──────────────────────────────────────────────────────────────────────────────
POSE_TARGETS: dict[str, TargetDef] = {
    # Pose Facial Fallbacks
    "pose nose":      TargetDef("Pose Nose",      TrackerType.POSE, 0,  emoji="👃"),
    "pose left eye":  TargetDef("Pose L. Eye",    TrackerType.POSE, 1,  emoji="👁️"),
    "pose right eye": TargetDef("Pose R. Eye",    TrackerType.POSE, 2,  emoji="👁️"),
    "left ear":       TargetDef("L. Ear",         TrackerType.POSE, 3,  emoji="👂"),
    "right ear":      TargetDef("R. Ear",         TrackerType.POSE, 4,  emoji="👂"),
    
    # Upper Body
    "left shoulder":  TargetDef("L. Shoulder",    TrackerType.POSE, 5,  emoji="💪"),
    "right shoulder": TargetDef("R. Shoulder",    TrackerType.POSE, 6,  emoji="💪"),
    "shoulder":       TargetDef("R. Shoulder",    TrackerType.POSE, 6,  emoji="💪"),
    "left elbow":     TargetDef("L. Elbow",       TrackerType.POSE, 7,  emoji="🦾"),
    "right elbow":    TargetDef("R. Elbow",       TrackerType.POSE, 8,  emoji="🦾"),
    "elbow":          TargetDef("R. Elbow",       TrackerType.POSE, 8,  emoji="🦾"),
    "left wrist":     TargetDef("L. Wrist",       TrackerType.POSE, 9,  emoji="✋"),
    "right wrist":    TargetDef("R. Wrist",       TrackerType.POSE, 10, emoji="✋"),
    "wrist":          TargetDef("R. Wrist",       TrackerType.POSE, 10, emoji="✋"),
    
    # Lower Body
    "left hip":       TargetDef("L. Hip",         TrackerType.POSE, 11, emoji="🦵"),
    "right hip":      TargetDef("R. Hip",         TrackerType.POSE, 12, emoji="🦵"),
    "hip":            TargetDef("R. Hip",         TrackerType.POSE, 12, emoji="🦵"),
    "left knee":      TargetDef("L. Knee",        TrackerType.POSE, 13, emoji="🦵"),
    "right knee":     TargetDef("R. Knee",        TrackerType.POSE, 14, emoji="🦵"),
    "knee":           TargetDef("R. Knee",        TrackerType.POSE, 14, emoji="🦵"),
    "left ankle":     TargetDef("L. Ankle",       TrackerType.POSE, 15, emoji="🦶"),
    "right ankle":    TargetDef("R. Ankle",       TrackerType.POSE, 16, emoji="🦶"),
    "ankle":          TargetDef("R. Ankle",       TrackerType.POSE, 16, emoji="🦶"),
}

# Merged lookup (FACE takes priority for overlapping keywords like "nose")
ALL_TARGETS: dict[str, TargetDef] = {**POSE_TARGETS, **FACE_TARGETS}


def resolve(text: str) -> Optional[TargetDef]:
    """
    Resolve a user text string to a TargetDef.
    Case-insensitive, strips extra whitespace.
    Returns None if no match found.
    """
    key = text.strip().lower()
    return ALL_TARGETS.get(key)


def all_keywords() -> list[str]:
    """Return a sorted list of all valid target keywords."""
    return sorted(ALL_TARGETS.keys())
