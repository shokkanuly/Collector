"""HandPose -> ServoFrame using config/hand.yaml (ARCHITECTURE.md §4.3).

Applies calibration (min_deg/max_deg), inversion, rounding, and clamping. This
is the Python half of the two-layer servo safety (CLAUDE.md rule 5).

Implemented in ROADMAP stage 2.
"""
