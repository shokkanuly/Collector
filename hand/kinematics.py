"""Landmarks -> HandPose (ARCHITECTURE.md §4.1).

Every function is pure and vectorized. It accepts one hand as the 63 floats
utils/hand_tracker.py produces, or a batch shaped (N, 63), and returns a float
(a (4,) array for curl) for one hand or an array for a batch. build_poses.py and
the tests run it over the whole dataset; mirror mode runs it on single frames.

Two layers:
- raw measurements in physical units: degrees, and distances in the tracker's
  palm unit (|wrist -> landmark 9| = 1);
- normalization to HandPose fields using ranges frozen from the dataset, so a
  live frame maps exactly the way the dataset did. build_poses.py recomputes
  the ranges (dataset_ranges) and reports any drift.

MediaPipe's x and y are fractions of the image width and height, so on a 4:3 or
16:9 camera the hand is stretched slightly along x, and every angle here
inherits that. The dataset and the live demo share a camera, so the frozen
ranges absorb it; a camera with a different aspect ratio shifts values a little.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .types import FINGER_LANDMARKS, FINGERS, N_FEATURES, N_LANDMARKS, WRIST, HandPose


@dataclass(frozen=True)
class Range:
    """Maps a raw measurement linearly onto 0..1 (`lo` -> 0, `hi` -> 1), clipped.

    A range with lo > hi maps inversely; thumb opposition uses that.
    """

    lo: float
    hi: float

    def to_01(self, raw):
        return np.clip((np.asarray(raw, dtype=float) - self.lo) / (self.hi - self.lo), 0.0, 1.0)


# ---------------------------------------------------------------- frozen normalization
# 2nd/98th percentiles over landmarks_dataset.csv (3,765 rows, 24 letters),
# recomputed by dataset_ranges().
#
# Curl ranges are per finger. MediaPipe's index never reads as bent as the other
# fingers in a fist (98th percentile 232 deg against 302-317), and each finger
# has its own servo, so curl_01 = 1 has to mean "this finger's full fist".
CURL_RANGE_DEG = {
    "index": Range(9.4, 231.7),
    "middle": Range(7.3, 317.2),
    "ring": Range(7.2, 309.2),
    "pinky": Range(8.0, 301.9),
}
THUMB_FLEX_RANGE_DEG = Range(14.0, 133.6)
# Thumb tip (4) to pinky MCP (17): far (thumb out, L) -> 0, close (across the palm, M) -> 1.
THUMB_OPP_RANGE = Range(1.82, 0.21)
# Taken only over frames where index and middle are both extended. Elsewhere the
# angle between their proximal phalanges measures flexion, not abduction
# (G, P, Q, X read 70-120 deg), and would swamp the range.
SPREAD_RANGE_DEG = Range(1.1, 17.6)
EXTENDED_CURL_01 = 0.3
# Spread fades from measured to 0 as the more-curled of index/middle goes from
# curl_01 0.3 to 0.6: abduction of a curled finger is undefined, and 0 (together)
# is how fingers sit in a fist.
SPREAD_FADE_CURL_01 = (0.3, 0.6)
WRIST_ROLL_FULL_DEG = 90.0  # palm edge-on to the camera -> wrist_roll = +-1


def frozen_ranges() -> Dict[str, Range]:
    """The constants above, keyed like dataset_ranges()."""
    ranges = {f"curl.{f}": CURL_RANGE_DEG[f] for f in FINGERS}
    ranges.update(thumb_flex=THUMB_FLEX_RANGE_DEG, thumb_opp=THUMB_OPP_RANGE, spread=SPREAD_RANGE_DEG)
    return ranges


# ---------------------------------------------------------------- geometry helpers

def as_points(landmarks) -> np.ndarray:
    """(63,) or (N, 63) landmarks -> (N, 21, 3) points."""
    arr = np.asarray(landmarks, dtype=float)
    if arr.ndim not in (1, 2) or arr.shape[-1] != N_FEATURES:
        raise ValueError(f"expected landmarks shaped (63,) or (N, 63), got {arr.shape}")
    return arr.reshape(-1, N_LANDMARKS, 3)


def _one_or_many(landmarks, values: np.ndarray):
    """Drop the batch axis again when the caller passed a single hand."""
    if np.ndim(landmarks) == 1:
        return values[0] if values.ndim > 1 else float(values[0])
    return values


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, norm, out=np.zeros_like(v), where=norm > 0)


def _angle_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # atan2(|u x v|, u . v) stays accurate near 0 and 180 deg, where arccos of a
    # rounded cosine does not (a straight finger would read ~1e-6 deg). A
    # zero-length segment gives 0.
    cross = np.linalg.norm(np.cross(u, v), axis=-1)
    return np.degrees(np.arctan2(cross, np.sum(u * v, axis=-1)))


def _bend_deg(points: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    """Bend at joint b between segments a->b and b->c; 0 = straight."""
    return _angle_deg(points[:, b] - points[:, a], points[:, c] - points[:, b])


# ---------------------------------------------------------------- raw measurements

def curl_deg(landmarks):
    """Bend summed over MCP, PIP, and DIP for index, middle, ring, pinky.

    The MCP bend is measured against the wrist -> MCP line. Returns (4,) or (N, 4).
    """
    p = as_points(landmarks)
    per_finger = [_bend_deg(p, WRIST, mcp, pip) + _bend_deg(p, mcp, pip, dip) + _bend_deg(p, pip, dip, tip)
                  for mcp, pip, dip, tip in (FINGER_LANDMARKS[f] for f in FINGERS)]
    return _one_or_many(landmarks, np.stack(per_finger, axis=-1))


def thumb_flex_deg(landmarks):
    """Bend at the thumb MCP (landmark 2) plus bend at the IP joint (landmark 3)."""
    p = as_points(landmarks)
    return _one_or_many(landmarks, _bend_deg(p, 1, 2, 3) + _bend_deg(p, 2, 3, 4))


def thumb_pinky_dist(landmarks):
    """Distance from the thumb tip (4) to the pinky MCP (17), in palm units."""
    p = as_points(landmarks)
    return _one_or_many(landmarks, np.linalg.norm(p[:, 4] - p[:, 17], axis=-1))


def spread_deg(landmarks):
    """Angle between the index (5->6) and middle (9->10) proximal phalanges."""
    p = as_points(landmarks)
    return _one_or_many(landmarks, _angle_deg(p[:, 6] - p[:, 5], p[:, 10] - p[:, 9]))


def palm_normal(landmarks):
    """Unit normal of the palm plane through the wrist and the index and pinky MCPs."""
    p = as_points(landmarks)
    return _one_or_many(landmarks, _unit(np.cross(p[:, 5] - p[:, WRIST], p[:, 17] - p[:, WRIST])))


def wrist_roll_deg(landmarks):
    """Forearm roll from where the palm faces: 0 = toward the camera, +90 = image right.

    HandPose defines roll by where the palm faces. The in-plane angle of
    wrist -> landmark 9 that ARCHITECTURE §4.1 first proposed measures tilt
    within the image instead, and cannot see roll at all.
    """
    n = np.atleast_2d(palm_normal(landmarks))
    return _one_or_many(landmarks, np.degrees(np.arctan2(n[:, 0], n[:, 2])))


# ---------------------------------------------------------------- HandPose fields

def curl_01(landmarks):
    """Per-finger curl, 0 = straight, 1 = this finger's full fist. (4,) or (N, 4)."""
    deg = np.atleast_2d(curl_deg(landmarks))
    out = np.stack([CURL_RANGE_DEG[f].to_01(deg[:, i]) for i, f in enumerate(FINGERS)], axis=-1)
    return _one_or_many(landmarks, out)


def thumb_flex_01(landmarks):
    return _one_or_many(landmarks, THUMB_FLEX_RANGE_DEG.to_01(np.atleast_1d(thumb_flex_deg(landmarks))))


def thumb_opp_01(landmarks):
    """0 = thumb out to the side (L), 1 = across the palm toward the pinky (M)."""
    return _one_or_many(landmarks, THUMB_OPP_RANGE.to_01(np.atleast_1d(thumb_pinky_dist(landmarks))))


def _spread_01(spread: np.ndarray, curl: np.ndarray) -> np.ndarray:
    lo, hi = SPREAD_FADE_CURL_01
    more_curled = np.maximum(curl[:, 0], curl[:, 1])  # index, middle
    fade = np.clip((hi - more_curled) / (hi - lo), 0.0, 1.0)
    return SPREAD_RANGE_DEG.to_01(spread) * fade


def spread_01(landmarks):
    """Index-middle abduction, 0 = together (U), 1 = wide (V); 0 once either finger curls."""
    spread = np.atleast_1d(spread_deg(landmarks))
    return _one_or_many(landmarks, _spread_01(spread, np.atleast_2d(curl_01(landmarks))))


def wrist_roll(landmarks):
    """-1..1: wrist_roll_deg scaled so +-90 deg (palm edge-on) is +-1."""
    deg = np.atleast_1d(wrist_roll_deg(landmarks))
    return _one_or_many(landmarks, np.clip(deg / WRIST_ROLL_FULL_DEG, -1.0, 1.0))


def pose_features(landmarks) -> Dict[str, np.ndarray]:
    """Every HandPose field for a batch, as arrays: curl (N, 4), the rest (N,)."""
    batch = np.atleast_2d(np.asarray(landmarks, dtype=float))
    curl = curl_01(batch)
    return {
        "curl": curl,
        "thumb_flex": thumb_flex_01(batch),
        "thumb_opp": thumb_opp_01(batch),
        "spread": _spread_01(spread_deg(batch), curl),
        "wrist_roll": wrist_roll(batch),
    }


def pose_from_landmarks(landmarks) -> HandPose:
    """One hand (63 floats) -> HandPose."""
    arr = np.asarray(landmarks, dtype=float)
    if arr.shape != (N_FEATURES,):
        raise ValueError(f"expected one hand of {N_FEATURES} floats, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("landmarks contain NaN or inf")
    f = pose_features(arr)
    return HandPose(curl=tuple(f["curl"][0]), thumb_flex=f["thumb_flex"][0],
                    thumb_opp=f["thumb_opp"][0], spread=f["spread"][0],
                    wrist_roll=f["wrist_roll"][0])


# ---------------------------------------------------------------- recomputing the ranges

def dataset_ranges(landmarks, lo_pct: float = 2.0, hi_pct: float = 98.0) -> Dict[str, Range]:
    """Recompute every normalization range from a batch, the way the constants were made."""
    batch = np.atleast_2d(np.asarray(landmarks, dtype=float))
    curl = curl_deg(batch)
    ranges = {}
    for i, finger in enumerate(FINGERS):
        lo, hi = np.percentile(curl[:, i], [lo_pct, hi_pct])
        ranges[f"curl.{finger}"] = Range(float(lo), float(hi))
    lo, hi = np.percentile(thumb_flex_deg(batch), [lo_pct, hi_pct])
    ranges["thumb_flex"] = Range(float(lo), float(hi))
    lo, hi = np.percentile(thumb_pinky_dist(batch), [lo_pct, hi_pct])
    ranges["thumb_opp"] = Range(float(hi), float(lo))  # inverse: far = 0
    extended = ((ranges["curl.index"].to_01(curl[:, 0]) < EXTENDED_CURL_01)
                & (ranges["curl.middle"].to_01(curl[:, 1]) < EXTENDED_CURL_01))
    lo, hi = np.percentile(spread_deg(batch)[extended], [lo_pct, hi_pct])
    ranges["spread"] = Range(float(lo), float(hi))
    return ranges
