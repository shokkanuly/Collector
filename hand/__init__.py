"""Collector Hand: the Python side of the robotic hand.

Design: docs/hardware/ARCHITECTURE.md §4. Everything under hand/ is optional
(CLAUDE.md rule 1): importing this package stays cheap and never imports
pyserial. Hardware-facing modules load their dependencies lazily.
"""
from .types import JOINTS, N_CH, HandPose, ServoFrame

__all__ = ["JOINTS", "N_CH", "HandPose", "ServoFrame"]
