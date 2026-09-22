"""HandPose -> ServoFrame using config/hand.yaml (ARCHITECTURE.md §4.3).

For every channel: angle = min_deg + value_01 * (max_deg - min_deg), with
value_01 flipped first when the channel is inverted, then rounded and clamped
to [min_deg, max_deg]. That clamp is the Python half of the two-layer servo
safety; the firmware clamps again to its own compiled-in limits (CLAUDE.md
rule 5).

wrist_roll is the only field outside 0..1: its -1..1 maps to the whole
channel range, so 0 (palm to the viewer) lands mid-range.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from .types import JOINTS, N_CH, SERVO_MAX_DEG, SERVO_MIN_DEG, HandPose, ServoFrame

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "hand.yaml")


@dataclass(frozen=True)
class ChannelConfig:
    """One PCA9685 channel: the joint it drives and its calibrated range."""

    ch: int
    joint: str
    min_deg: int
    max_deg: int
    invert: bool = False

    def __post_init__(self) -> None:
        if self.joint not in JOINTS:
            raise ValueError(f"channel {self.ch}: unknown joint {self.joint!r}; expected one of {JOINTS}")
        for name in ("ch", "min_deg", "max_deg"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"channel {self.ch}: {name} must be an integer, got {value!r}")
        if not 0 <= self.ch < N_CH:
            raise ValueError(f"channel {self.ch} is outside 0-{N_CH - 1}")
        if not SERVO_MIN_DEG <= self.min_deg <= self.max_deg <= SERVO_MAX_DEG:
            raise ValueError(f"channel {self.ch} ({self.joint}): need {SERVO_MIN_DEG} <= min_deg <= max_deg "
                             f"<= {SERVO_MAX_DEG}, got {self.min_deg}..{self.max_deg} "
                             "(use `invert` to reverse a direction)")
        if not isinstance(self.invert, bool):
            raise ValueError(f"channel {self.ch}: invert must be true or false, got {self.invert!r}")


@dataclass(frozen=True)
class HandConfig:
    """Everything in config/hand.yaml."""

    port: str
    baud: int
    rate_hz: float
    channels: Tuple[ChannelConfig, ...]
    letter_hold_ms: int
    smoothing_alpha: float

    def __post_init__(self) -> None:
        if not self.rate_hz > 0:
            raise ValueError(f"rate_hz must be positive, got {self.rate_hz}")
        if not 0 < self.smoothing_alpha <= 1:
            raise ValueError(f"smoothing_alpha must be in (0, 1], got {self.smoothing_alpha}")
        if self.letter_hold_ms < 0:
            raise ValueError(f"letter_hold_ms must be >= 0, got {self.letter_hold_ms}")


def config_from_dict(raw: Mapping[str, Any]) -> HandConfig:
    channels = tuple(ChannelConfig(ch=c["ch"], joint=c["joint"], min_deg=c["min_deg"],
                                   max_deg=c["max_deg"], invert=c.get("invert", False))
                     for c in raw["channels"])
    return HandConfig(port=str(raw.get("port", "auto")), baud=int(raw.get("baud", 115200)),
                      rate_hz=float(raw.get("rate_hz", 30)), channels=channels,
                      letter_hold_ms=int(raw.get("letter_hold_ms", 900)),
                      smoothing_alpha=float(raw.get("smoothing_alpha", 0.35)))


def load_config(path: str = DEFAULT_CONFIG) -> HandConfig:
    import yaml  # lazy: only the hardware path needs it

    with open(path) as f:
        return config_from_dict(yaml.safe_load(f))


class ServoMapper:
    """Turns a HandPose into calibrated, clamped servo angles."""

    def __init__(self, channels: Sequence[ChannelConfig]):
        chans = sorted(channels, key=lambda c: c.ch)
        if [c.ch for c in chans] != list(range(N_CH)):
            raise ValueError(f"need exactly one entry for each channel 0-{N_CH - 1}, got {[c.ch for c in chans]}")
        if sorted(c.joint for c in chans) != sorted(JOINTS):
            raise ValueError(f"every joint needs exactly one channel; got {[c.joint for c in chans]}")
        self.channels: Tuple[ChannelConfig, ...] = tuple(chans)

    @classmethod
    def from_config(cls, config: HandConfig) -> "ServoMapper":
        return cls(config.channels)

    @staticmethod
    def angle_deg(channel: ChannelConfig, value_01: float) -> int:
        """Map a 0..1 value onto one channel's calibrated range."""
        v = min(max(float(value_01), 0.0), 1.0)
        if channel.invert:
            v = 1.0 - v
        angle = math.floor(channel.min_deg + v * (channel.max_deg - channel.min_deg) + 0.5)
        return min(max(angle, channel.min_deg), channel.max_deg)

    def to_frame(self, pose: HandPose) -> ServoFrame:
        angles = []
        for channel in self.channels:
            value = pose.joint_value(channel.joint)
            if channel.joint == "wrist_roll":
                value = (value + 1.0) / 2.0  # -1..1 -> 0..1
            angles.append(self.angle_deg(channel, value))
        return ServoFrame(tuple(angles))
