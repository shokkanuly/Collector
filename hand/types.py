"""Core value types shared by every hand module (docs/hardware/ARCHITECTURE.md §3).

`HandPose` is the seam between the software and hardware halves: everything
upstream (dataset, recognizer letter, mirror mode) produces one, and everything
downstream (mapper, link, firmware) consumes one. It is normalized and knows
nothing about servos. `ServoFrame` is the hardware-specific result after
calibration, one integer angle per PCA9685 channel.

This module shares its name with the stdlib `types`. That only matters if a
file inside hand/ is run directly as a script, which none of them are meant to be.
"""
from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

# Servo channels driven on the PCA9685 (PROTOCOL.md: "READY v1 ch=8").
N_CH = 8
SERVO_MIN_DEG = 0
SERVO_MAX_DEG = 180

# Joint driven by each channel, in channel order (config/hand.yaml, WIRING.md §2.4).
JOINTS = ("thumb_flex", "thumb_opp", "index", "middle", "ring", "pinky", "spread", "wrist_roll")
FINGERS = ("index", "middle", "ring", "pinky")

# MediaPipe hand landmarks. utils/hand_tracker.py emits 21 points x (x, y, z)
# flattened to 63 floats, translated to the wrist and scaled by |wrist -> 9|.
N_LANDMARKS = 21
N_FEATURES = 63
WRIST = 0
THUMB = (1, 2, 3, 4)       # CMC, MCP, IP, tip
INDEX = (5, 6, 7, 8)       # MCP, PIP, DIP, tip
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGER_LANDMARKS = {"index": INDEX, "middle": MIDDLE, "ring": RING, "pinky": PINKY}


def _unit_float(name: str, value: Any, lo: float, hi: float) -> float:
    value = float(value)
    if not (math.isfinite(value) and lo <= value <= hi):
        raise ValueError(f"{name}={value!r} is outside [{lo}, {hi}]")
    return value


@dataclass(frozen=True)
class HandPose:
    """A hardware-independent hand pose. Every field is normalized.

    curl        index, middle, ring, pinky; 0 = straight, 1 = full fist
    thumb_flex  0 = straight, 1 = fully bent
    thumb_opp   0 = thumb out to the side (L, Y), 1 = across the palm toward
                the pinky (M, B). A's thumb, beside the index, sits in between.
    spread      index-middle abduction; 0 = together (U), 1 = wide (V)
    wrist_roll  -1 .. 1; 0 = palm facing the viewer, +-1 = +-90 degrees
    """

    curl: Tuple[float, float, float, float]
    thumb_flex: float
    thumb_opp: float
    spread: float
    wrist_roll: float = 0.0

    def __post_init__(self) -> None:
        # Accept any 4-sequence (a JSON list, numpy floats) but store plain
        # floats in a tuple so a pose is immutable, hashable, and serializable.
        curl = tuple(self.curl)
        if len(curl) != len(FINGERS):
            raise ValueError(f"curl needs {len(FINGERS)} values (index, middle, ring, pinky), got {len(curl)}")
        object.__setattr__(self, "curl", tuple(
            _unit_float(f"curl[{finger}]", c, 0.0, 1.0) for finger, c in zip(FINGERS, curl)))
        for name in ("thumb_flex", "thumb_opp", "spread"):
            object.__setattr__(self, name, _unit_float(name, getattr(self, name), 0.0, 1.0))
        object.__setattr__(self, "wrist_roll", _unit_float("wrist_roll", self.wrist_roll, -1.0, 1.0))

    def joint_value(self, joint: str) -> float:
        """The value that drives `joint` (a name from JOINTS), in the field's own range."""
        if joint in FINGERS:
            return self.curl[FINGERS.index(joint)]
        if joint in ("thumb_flex", "thumb_opp", "spread", "wrist_roll"):
            return getattr(self, joint)
        raise KeyError(f"unknown joint {joint!r}; expected one of {JOINTS}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "curl": list(self.curl),
            "thumb_flex": self.thumb_flex,
            "thumb_opp": self.thumb_opp,
            "spread": self.spread,
            "wrist_roll": self.wrist_roll,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HandPose":
        """Build a pose from `to_dict()` output. Extra keys (e.g. metadata) are ignored."""
        return cls(
            curl=tuple(data["curl"]),
            thumb_flex=data["thumb_flex"],
            thumb_opp=data["thumb_opp"],
            spread=data["spread"],
            wrist_roll=data.get("wrist_roll", 0.0),
        )


@dataclass(frozen=True)
class ServoFrame:
    """Target angles for all N_CH channels, already calibrated and clamped.

    One frame becomes one PROTOCOL.md `S` line.
    """

    angles_deg: Tuple[int, ...]

    def __post_init__(self) -> None:
        angles = tuple(self.angles_deg)
        if len(angles) != N_CH:
            raise ValueError(f"a ServoFrame needs {N_CH} angles, got {len(angles)}")
        checked = []
        for ch, angle in enumerate(angles):
            if isinstance(angle, bool):
                raise TypeError(f"channel {ch}: angle must be an int, got {angle!r}")
            angle = operator.index(angle)  # accepts numpy ints, rejects floats
            if not SERVO_MIN_DEG <= angle <= SERVO_MAX_DEG:
                raise ValueError(f"channel {ch}: {angle} deg is outside [{SERVO_MIN_DEG}, {SERVO_MAX_DEG}]")
            checked.append(angle)
        object.__setattr__(self, "angles_deg", tuple(checked))
