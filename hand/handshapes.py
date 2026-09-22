"""What each fingerspelled letter is supposed to look like, as categorical states.

The dataset measures what MediaPipe sees; this table says what the hand should
do. scripts/build_poses.py uses it to shape the dataset medians into natural
robot poses, and the tests check every pose in hand/poses.json against it.

Why shaping is needed: MediaPipe reads a finger tucked under the thumb (the ring
and pinky in U, V, K, R, ...) as only 55-80% closed, because the thumb hides it.
Taken literally, 42 of the 51 fingers that ASL closes would stop half-bent,
sticking straight out from the palm. So a finger that is extended or closed in
ASL gets exactly 0 or 1, and only fingers in a partial shape (the curves of C,
O, D, E, the fold of M and N, the hook of X) keep the measured curl.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .types import FINGERS, HandPose

# Finger states.
EXTENDED = "extended"
CLOSED = "closed"      # a fist, or tucked under the thumb
PARTIAL = "partial"    # curved (C, O), bent (E), folded over the thumb (M, N), hooked (X)

# Index-middle spread states.
TOGETHER = "together"
APART = "apart"

# Thumb states.
OUT = "out"            # abducted away from the hand (L, Y)
BESIDE = "beside"      # straight, along the side of the index (A)
PARALLEL = "parallel"  # extended next to the pointing index (G, Q)
BETWEEN = "between"    # between index and middle (K, P, T)
TOUCH = "touch"        # opposing the fingertips (C, D, F, O)
OVER = "over"          # across the palm or over closed fingers (B, S, U, ...)
UNDER = "under"        # tucked under bent fingers (E, M, N)

# Values a definitive state is shaped to. APART leaves headroom below the widest
# spread (OPEN, 1.0); a calibrated spread servo makes 0.8 a clear V.
CURL_TARGET = {EXTENDED: 0.0, CLOSED: 1.0}
SPREAD_TARGET = {TOGETHER: 0.0, APART: 0.8}

# What a finished pose must satisfy, per state: (low, high), inclusive.
CURL_BOUNDS = {EXTENDED: (0.0, 0.15), CLOSED: (0.85, 1.0), PARTIAL: (0.2, 0.92)}
SPREAD_BOUNDS = {TOGETHER: (0.0, 0.15), APART: (0.5, 1.0)}
THUMB_BOUNDS = {  # state: ((opp low, opp high), (flex low, flex high))
    OUT: ((0.0, 0.2), (0.0, 0.35)),
    BESIDE: ((0.4, 0.85), (0.0, 0.4)),
    PARALLEL: ((0.3, 0.8), (0.0, 0.35)),
    BETWEEN: ((0.5, 0.9), (0.0, 0.5)),
    TOUCH: ((0.5, 0.95), (0.05, 0.7)),
    OVER: ((0.75, 1.0), (0.2, 1.0)),
    UNDER: ((0.8, 1.0), (0.2, 1.0)),
}


@dataclass(frozen=True)
class Handshape:
    fingers: Tuple[str, str, str, str]  # index, middle, ring, pinky
    thumb: str
    spread: str
    note: str


def _shape(code: str, thumb: str, spread: str, note: str) -> Handshape:
    states = {"E": EXTENDED, "C": CLOSED, "P": PARTIAL}
    return Handshape(tuple(states[c] for c in code), thumb, spread, note)


# Standard one-handed ASL fingerspelling. Finger codes: E extended, C closed, P partial.
HANDSHAPES: Dict[str, Handshape] = {
    "A": _shape("CCCC", BESIDE, TOGETHER, "fist, thumb straight along the index"),
    "B": _shape("EEEE", OVER, TOGETHER, "flat hand, thumb folded across the palm"),
    "C": _shape("PPPP", TOUCH, TOGETHER, "all fingers curved into a C, thumb opposite"),
    "D": _shape("EPPP", TOUCH, TOGETHER, "index up; the others curve to meet the thumb"),
    "E": _shape("PPPP", UNDER, TOGETHER, "fingers bent down onto the tucked thumb"),
    "F": _shape("PEEE", TOUCH, TOGETHER, "index and thumb make a circle, three fingers up"),
    "G": _shape("ECCC", PARALLEL, TOGETHER, "index points sideways, thumb parallel"),
    "H": _shape("EECC", OVER, TOGETHER, "index and middle together, sideways"),
    "I": _shape("CCCE", OVER, TOGETHER, "pinky up, thumb over the fist"),
    "J": _shape("CCCE", OVER, TOGETHER, "I, then the pinky traces a hook"),
    "K": _shape("EECC", BETWEEN, APART, "index and middle apart, thumb between them"),
    "L": _shape("ECCC", OUT, TOGETHER, "index up, thumb out"),
    "M": _shape("PPPC", UNDER, TOGETHER, "three fingers folded over the thumb"),
    "N": _shape("PPCC", UNDER, TOGETHER, "two fingers folded over the thumb"),
    "O": _shape("PPPP", TOUCH, TOGETHER, "fingertips meet the thumb in an O"),
    "P": _shape("EPCC", BETWEEN, APART, "K pointing down; the middle angles forward"),
    "Q": _shape("ECCC", PARALLEL, TOGETHER, "G pointing down"),
    "R": _shape("EECC", OVER, TOGETHER, "index and middle crossed (no crossing joint: together)"),
    "S": _shape("CCCC", OVER, TOGETHER, "fist, thumb across the front of the fingers"),
    "T": _shape("PCCC", BETWEEN, TOGETHER, "fist, thumb between index and middle"),
    "U": _shape("EECC", OVER, TOGETHER, "index and middle up together"),
    "V": _shape("EECC", OVER, APART, "index and middle up in a V"),
    "W": _shape("EEEC", OVER, APART, "three fingers up and spread"),
    "X": _shape("PCCC", OVER, TOGETHER, "index hooked"),
    "Y": _shape("CCCE", OUT, TOGETHER, "thumb and pinky out"),
    "Z": _shape("ECCC", OVER, TOGETHER, "index points, then draws a Z"),
}

# Letters whose four fingers make a full fist: the curl range's "closed" anchor.
FIST_LETTERS = ("A", "S")


def shape_pose(letter: str, pose: Dict) -> List[str]:
    """Set every definitive finger and spread state of `pose` (a poses.json dict) in place.

    Returns the fields changed, e.g. ["curl.ring", "spread"].
    """
    spec = HANDSHAPES[letter]
    changed = []
    for i, (finger, state) in enumerate(zip(FINGERS, spec.fingers)):
        if state in CURL_TARGET and pose["curl"][i] != CURL_TARGET[state]:
            pose["curl"][i] = CURL_TARGET[state]
            changed.append(f"curl.{finger}")
    if pose["spread"] != SPREAD_TARGET[spec.spread]:
        pose["spread"] = SPREAD_TARGET[spec.spread]
        changed.append("spread")
    return changed


def check(letter: str, pose: HandPose) -> List[str]:
    """Every way `pose` departs from the letter's handshape; empty when it is natural."""
    spec = HANDSHAPES[letter]
    problems = []
    for finger, state, value in zip(FINGERS, spec.fingers, pose.curl):
        lo, hi = CURL_BOUNDS[state]
        if not lo <= value <= hi:
            problems.append(f"{letter} {finger} should be {state} ({lo}-{hi}), is {value:.2f}")
    lo, hi = SPREAD_BOUNDS[spec.spread]
    if not lo <= pose.spread <= hi:
        problems.append(f"{letter} spread should be {spec.spread} ({lo}-{hi}), is {pose.spread:.2f}")
    (opp_lo, opp_hi), (flex_lo, flex_hi) = THUMB_BOUNDS[spec.thumb]
    if not opp_lo <= pose.thumb_opp <= opp_hi:
        problems.append(f"{letter} thumb should be {spec.thumb} (opp {opp_lo}-{opp_hi}), is {pose.thumb_opp:.2f}")
    if not flex_lo <= pose.thumb_flex <= flex_hi:
        problems.append(f"{letter} thumb should be {spec.thumb} (flex {flex_lo}-{flex_hi}), is {pose.thumb_flex:.2f}")
    return problems


def recording_disagreements(letter: str, measured_curl: Tuple[float, ...]) -> List[str]:
    """Fingers whose recorded (pre-shaping) curl contradicts the handshape outright.

    These point at recordings to redo, not at the normal tucked-finger under-read:
    a closed finger must read below 0.5, or an extended one above 0.3, to count.
    """
    spec = HANDSHAPES[letter]
    out = []
    for finger, state, value in zip(FINGERS, spec.fingers, measured_curl):
        if state == EXTENDED and value > 0.3:
            out.append(f"{finger} should be extended but reads {value:.2f}")
        elif state == CLOSED and value < 0.5:
            out.append(f"{finger} should be closed but reads {value:.2f}")
        elif state == PARTIAL and value < 0.15:
            out.append(f"{finger} should be partly bent but reads {value:.2f} (straight)")
    return out
