"""Build hand/poses.json and reports/poses_report.md from the dataset (ARCHITECTURE.md §4.2).

    python scripts/build_poses.py

Reads landmarks_dataset.csv (never writes it; CLAUDE.md rule 3), runs
hand.kinematics over every row, takes each letter's median pose, applies the
hand-authored corrections in hand/pose_overrides.json field by field, then adds
the motion stubs J and Z and the named poses REST and OPEN.

hand/poses.json is the single pose source of truth (CLAUDE.md rule 4). Never
edit it by hand: change the overrides (or the dataset) and rebuild.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand import kinematics  # noqa: E402
from hand.types import FINGERS, LANDMARK_COLUMNS, HandPose  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(REPO_ROOT, "landmarks_dataset.csv")
OVERRIDES = os.path.join(REPO_ROOT, "hand", "pose_overrides.json")
POSES_JSON = os.path.join(REPO_ROOT, "hand", "poses.json")
REPORT = os.path.join(REPO_ROOT, "reports", "poses_report.md")

FIELDS = ("curl", "thumb_flex", "thumb_opp", "spread", "wrist_roll")
WIDE_IQR = 0.25  # flagged in the report: signers were inconsistent or the letter is ambiguous

# Named poses with no dataset source (PROTOCOL.md `H` goes to REST).
# REST: every tendon slack (each channel at its calibrated open end), thumb out,
# wrist neutral; the lowest-stress pose. OPEN: the "5" handshape, fingers spread.
NAMED_POSES = {
    "REST": HandPose(curl=(0, 0, 0, 0), thumb_flex=0, thumb_opp=0, spread=0, wrist_roll=0),
    "OPEN": HandPose(curl=(0, 0, 0, 0), thumb_flex=0, thumb_opp=0, spread=1, wrist_roll=0),
}
MOTION_NOTES = {
    "J": "J traces a hook with the pinky; this is its I handshape only",
    "Z": "Z draws a zigzag with the index; this is its index-point handshape only",
}


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _r3(value: float) -> float:
    return round(float(value), 3)


def letter_stats(features: Dict[str, np.ndarray], labels: np.ndarray,
                 sessions: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Per letter: row and session counts, and (median, q1, q3) for every field."""
    stats = {}
    for letter in sorted(set(labels)):
        rows = labels == letter
        entry: Dict[str, Any] = {"rows": int(rows.sum()), "sessions": len(set(sessions[rows]))}
        for field in FIELDS:
            values = features[field][rows]
            q1, med, q3 = np.percentile(values, [25, 50, 75], axis=0)
            entry[field] = (med, q1, q3)
        stats[letter] = entry
    return stats


def dataset_pose(entry: Dict[str, Any]) -> Dict[str, Any]:
    """The median pose of one letter. Measured wrist roll is not used for letters
    (ARCHITECTURE §4.1): orientation comes only from overrides."""
    return {
        "curl": [_r3(c) for c in entry["curl"][0]],
        "thumb_flex": _r3(entry["thumb_flex"][0]),
        "thumb_opp": _r3(entry["thumb_opp"][0]),
        "spread": _r3(entry["spread"][0]),
        "wrist_roll": 0.0,
    }


def apply_override(letter: str, pose: Dict[str, Any], override: Dict[str, Any]) -> List[str]:
    """Overwrite fields of `pose` in place; return the names of the fields changed.

    `curl` may be a full list of 4 or a {finger: value} dict for single fingers.
    """
    if not str(override.get("reason", "")).strip():
        raise ValueError(f"override for {letter} has no reason")
    changed = []
    for key, value in override.items():
        if key == "reason":
            continue
        if key == "curl" and isinstance(value, dict):
            for finger, v in value.items():
                pose["curl"][FINGERS.index(finger)] = _r3(v)
                changed.append(f"curl.{finger}")
        elif key == "curl":
            if len(value) != len(FINGERS):
                raise ValueError(f"override for {letter}: curl needs {len(FINGERS)} values")
            pose["curl"] = [_r3(v) for v in value]
            changed.append("curl")
        elif key in FIELDS:
            pose[key] = _r3(value)
            changed.append(key)
        else:
            raise ValueError(f"override for {letter}: unknown field {key!r}")
    return changed


def load_overrides(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}  # "_comment" keys


def build(dataset_path: str = DATASET, overrides_path: str = OVERRIDES,
          built: str = "") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (the poses.json document, the numbers the report is made from)."""
    df = pd.read_csv(dataset_path)
    landmarks = df[list(LANDMARK_COLUMNS)].to_numpy()
    labels = df["label"].to_numpy()
    features = kinematics.pose_features(landmarks)
    stats = letter_stats(features, labels, df["session_id"].to_numpy())

    poses: Dict[str, Dict[str, Any]] = {}
    for letter, entry in stats.items():
        poses[letter] = dict(dataset_pose(entry), source="dataset")

    overrides = load_overrides(overrides_path)
    for letter, override in sorted(overrides.items()):
        if letter not in poses:
            raise ValueError(f"override for {letter!r}, which the dataset does not have")
        changed = apply_override(letter, poses[letter], override)
        poses[letter].update(source="dataset+override", overridden=changed,
                             reason=override["reason"].strip())

    # Motion stubs, built from the final static poses (after overrides).
    poses["J"] = dict(poses["I"], source="motion-stub", motion=MOTION_NOTES["J"])
    for key in ("overridden", "reason"):
        poses["J"].pop(key, None)
    z = {field: poses["S"][field] for field in FIELDS}
    z["curl"] = [poses["D"]["curl"][0]] + poses["S"]["curl"][1:]  # D's index over S's fist
    z["spread"] = 0.0
    poses["Z"] = dict(z, source="motion-stub", motion=MOTION_NOTES["Z"])
    for name, pose in NAMED_POSES.items():
        poses[name] = dict(pose.to_dict(), source="named")

    for name, pose in poses.items():
        HandPose.from_dict(pose)  # every pose must be a valid HandPose

    frozen = kinematics.frozen_ranges()
    doc = {
        "_comment": "Generated by scripts/build_poses.py. Do not edit: change "
                    "hand/pose_overrides.json or the dataset and rebuild.",
        "version": 1,
        "built": built or datetime.date.today().isoformat(),
        "dataset": {"path": "landmarks_dataset.csv", "sha256": sha256_of(dataset_path),
                    "rows": int(len(df)), "letters": len(stats)},
        "normalization": {key: [r.lo, r.hi] for key, r in frozen.items()},
        "poses": poses,
    }
    numbers = {"stats": stats, "features": features, "labels": labels, "overrides": overrides,
               "recomputed": kinematics.dataset_ranges(landmarks), "frozen": frozen}
    return doc, numbers


# ---------------------------------------------------------------- report

def _cell(stat, i=None) -> str:
    med, q1, q3 = (s if i is None else s[i] for s in stat)
    flag = " ⚠" if q3 - q1 > WIDE_IQR else ""
    return f"{med:.2f} [{q1:.2f}–{q3:.2f}]{flag}"


def gate_rows(poses: Dict[str, Dict[str, Any]], stats: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, bool]]:
    """ROADMAP stage 2 "done when", checked on the dataset medians (before overrides)."""
    idx = {letter: float(stats[letter]["curl"][0][0]) for letter in stats}
    spread = {letter: float(stats[letter]["spread"][0]) for letter in stats}
    open_ok = all(idx[x] < 0.15 for x in "BWUV")
    fist_ok = all(idx[x] > 0.6 for x in "AST")
    diff = spread["V"] - spread["U"]
    names = set(poses)
    expected = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | {"REST", "OPEN"}
    return [
        ("B/W/U/V index curl < 0.15", ", ".join(f"{x} {idx[x]:.3f}" for x in "BWUV"), open_ok),
        ("A/S/T index curl > 0.6", ", ".join(f"{x} {idx[x]:.3f}" for x in "AST"), fist_ok),
        ("V − U spread ≥ 0.3", f"{spread['V']:.3f} − {spread['U']:.3f} = {diff:.3f}", diff >= 0.3),
        ("poses.json has A–Z + REST + OPEN", f"{len(names & expected)}/{len(expected)}", expected <= names),
    ]


def render_report(doc: Dict[str, Any], numbers: Dict[str, Any]) -> str:
    stats, poses = numbers["stats"], doc["poses"]
    ds = doc["dataset"]
    lines = [
        "# Pose library report",
        "",
        f"Generated by `scripts/build_poses.py` on {doc['built']} from `{ds['path']}` "
        f"({ds['rows']:,} rows, {ds['letters']} letters, sha256 `{ds['sha256'][:12]}…`). "
        "Do not edit; rebuild instead.",
        "",
        "## Stage-2 gate",
        "",
        "| Check | Value | Result |",
        "|---|---|---|",
    ]
    for check, value, ok in gate_rows(poses, stats):
        lines.append(f"| {check} | {value} | {'PASS' if ok else '**FAIL**'} |")

    lines += [
        "",
        "## Per-letter medians from the dataset (before overrides)",
        "",
        "Median over every frame of the letter, IQR in brackets; ⚠ marks an IQR wider than "
        f"{WIDE_IQR}. Curl: 0 = straight, 1 = fist. Thumb opp: 0 = out (L), 1 = across the palm (M). "
        "Spread: 0 = together (U), 1 = wide. Roll is measured from the palm normal "
        "(0 = palm to camera, ±1 = edge-on) and shown for information only: letters take "
        "orientation from overrides.",
        "",
        "| Letter | Sessions | Index | Middle | Ring | Pinky | Thumb flex | Thumb opp | Spread | Roll |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for letter, s in stats.items():
        curls = " | ".join(_cell(s["curl"], i) for i in range(len(FINGERS)))
        lines.append(f"| {letter} | {s['sessions']} | {curls} | {_cell(s['thumb_flex'])} | "
                     f"{_cell(s['thumb_opp'])} | {_cell(s['spread'])} | {s['wrist_roll'][0]:+.2f} |")

    lines += ["", "## Overrides applied (`hand/pose_overrides.json`)", ""]
    if numbers["overrides"]:
        lines += ["| Letter | Fields | Reason |", "|---|---|---|"]
        for letter in sorted(numbers["overrides"]):
            p = poses[letter]
            lines.append(f"| {letter} | {', '.join(p['overridden'])} | {p['reason']} |")
    else:
        lines.append("None.")

    lines += [
        "",
        "## Final poses (`hand/poses.json`)",
        "",
        "| Pose | Index | Middle | Ring | Pinky | Thumb flex | Thumb opp | Spread | Roll | Source |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(poses, key=lambda n: (len(n) > 1, n)):
        p = poses[name]
        curls = " | ".join(f"{c:.2f}" for c in p["curl"])
        lines.append(f"| {name} | {curls} | {p['thumb_flex']:.2f} | {p['thumb_opp']:.2f} | "
                     f"{p['spread']:.2f} | {p['wrist_roll']:+.2f} | {p['source']} |")

    lines += [
        "",
        "## Normalization ranges",
        "",
        "Frozen in `hand/kinematics.py`; recomputed here from the dataset (2nd/98th percentiles). "
        "`test_kinematics` fails if they drift apart by more than 2% of the range.",
        "",
        "| Range | Frozen | Recomputed |",
        "|---|---|---|",
    ]
    for key, frozen in numbers["frozen"].items():
        rec = numbers["recomputed"][key]
        lines.append(f"| {key} | {frozen.lo:.2f} → {frozen.hi:.2f} | {rec.lo:.2f} → {rec.hi:.2f} |")
    return "\n".join(lines) + "\n"


def write_outputs(doc: Dict[str, Any], report: str, poses_path: str = POSES_JSON,
                  report_path: str = REPORT) -> None:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(poses_path, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(report_path, "w") as f:
        f.write(report)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args(argv)
    doc, numbers = build()
    report = render_report(doc, numbers)
    write_outputs(doc, report)
    print(f"Wrote {os.path.relpath(POSES_JSON, REPO_ROOT)} ({len(doc['poses'])} poses) "
          f"and {os.path.relpath(REPORT, REPO_ROOT)}")
    failed = [check for check, _, ok in gate_rows(doc["poses"], numbers["stats"]) if not ok]
    for check in failed:
        print(f"GATE FAIL: {check}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
