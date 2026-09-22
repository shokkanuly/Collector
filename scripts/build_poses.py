"""Build hand/poses.json and reports/poses_report.md from the dataset (ARCHITECTURE.md §4.2).

    python scripts/build_poses.py

Reads landmarks_dataset.csv (never writes it; CLAUDE.md rule 3), runs
hand.kinematics over every row, and takes each letter's median pose. Then:

1. shape: fingers that ASL extends or closes get exactly 0 or 1, and the
   index-middle spread gets its together/apart value (hand/handshapes.py).
   MediaPipe reads tucked fingers as half-closed, so the raw medians would
   leave them sticking out; partial shapes keep the measured curl.
2. override: hand-authored corrections from hand/pose_overrides.json, field by
   field. A value may name another letter to copy that letter's shaped value.
3. add the motion stubs J and Z and the named poses REST and OPEN.

hand/poses.json is the single pose source of truth (CLAUDE.md rule 4). Never
edit it by hand: change the overrides, the handshapes, or the dataset, and rebuild.
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hand import handshapes, kinematics  # noqa: E402
from hand.types import FINGERS, LANDMARK_COLUMNS, HandPose  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(REPO_ROOT, "landmarks_dataset.csv")
OVERRIDES = os.path.join(REPO_ROOT, "hand", "pose_overrides.json")
POSES_JSON = os.path.join(REPO_ROOT, "hand", "poses.json")
REPORT = os.path.join(REPO_ROOT, "reports", "poses_report.md")

FIELDS = ("curl", "thumb_flex", "thumb_opp", "spread", "wrist_roll")
WIDE_IQR = 0.25  # flagged in the report: signers were inconsistent or the letter is ambiguous

# Named poses with no dataset source.
# REST (PROTOCOL.md `H`, and idle between words): a relaxed hand, with the
# fingers slightly bent and more so toward the pinky (the natural resting
# cascade), the thumb relaxed between out and beside, and every tendon near
# slack. A dead-flat hand looks stiff. OPEN: the "5" handshape, all fingers
# straight and spread, thumb out.
NAMED_POSES = {
    "REST": HandPose(curl=(0.12, 0.16, 0.21, 0.26), thumb_flex=0.15, thumb_opp=0.35, spread=0.2, wrist_roll=0),
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


def apply_override(letter: str, pose: Dict[str, Any], override: Dict[str, Any],
                   sources: Dict[str, Dict[str, Any]]) -> List[str]:
    """Overwrite fields of `pose` in place; return what changed, e.g. ["thumb_opp <- L"].

    `curl` may be a full list of 4 or a {finger: value} dict for single fingers.
    A string value names a letter in `sources` to copy that field from.
    """
    if not str(override.get("reason", "")).strip():
        raise ValueError(f"override for {letter} has no reason")

    def resolve(field: str, value: Any, finger: int = -1) -> Tuple[float, str]:
        if isinstance(value, str):
            ref = sources.get(value.upper())
            if ref is None:
                raise ValueError(f"override for {letter}: {value!r} is not a letter to copy from")
            copied = ref["curl"][finger] if field == "curl" else ref[field]
            return _r3(copied), f" <- {value.upper()}"
        return _r3(value), ""

    changed = []
    for key, value in override.items():
        if key == "reason":
            continue
        if key == "curl" and isinstance(value, dict):
            for finger, v in value.items():
                i = FINGERS.index(finger)
                pose["curl"][i], note = resolve("curl", v, i)
                changed.append(f"curl.{finger}{note}")
        elif key == "curl":
            if len(value) != len(FINGERS):
                raise ValueError(f"override for {letter}: curl needs {len(FINGERS)} values")
            pose["curl"] = [resolve("curl", v, i)[0] for i, v in enumerate(value)]
            changed.append("curl")
        elif key in FIELDS:
            pose[key], note = resolve(key, value)
            changed.append(f"{key}{note}")
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
    disagreements: Dict[str, List[str]] = {}
    for letter, entry in stats.items():
        pose = dataset_pose(entry)
        found = handshapes.recording_disagreements(letter, pose["curl"])
        if found:
            disagreements[letter] = found
        shaped = handshapes.shape_pose(letter, pose)
        poses[letter] = dict(pose, source="dataset", shaped=shaped)

    overrides = load_overrides(overrides_path)
    sources = copy.deepcopy(poses)  # overrides copy shaped values, never each other's
    for letter, override in sorted(overrides.items()):
        if letter not in poses:
            raise ValueError(f"override for {letter!r}, which the dataset does not have")
        changed = apply_override(letter, poses[letter], override, sources)
        poses[letter].update(overridden=changed, reason=override["reason"].strip())

    # Motion stubs, built from the final static poses (after overrides).
    poses["J"] = {field: copy.deepcopy(poses["I"][field]) for field in FIELDS}
    poses["Z"] = {field: copy.deepcopy(poses["S"][field]) for field in FIELDS}  # S's thumb over the fist
    for letter in ("J", "Z"):
        shaped = handshapes.shape_pose(letter, poses[letter])
        poses[letter].update(source="motion-stub", motion=MOTION_NOTES[letter], shaped=shaped)
    for name, pose in NAMED_POSES.items():
        poses[name] = dict(pose.to_dict(), source="named")

    for name, pose in poses.items():
        HandPose.from_dict(pose)  # every pose must be a valid HandPose

    frozen = kinematics.frozen_ranges()
    doc = {
        "_comment": "Generated by scripts/build_poses.py. Do not edit: change "
                    "hand/pose_overrides.json, hand/handshapes.py, or the dataset and rebuild.",
        "version": 2,
        "built": built or datetime.date.today().isoformat(),
        "dataset": {"path": "landmarks_dataset.csv", "sha256": sha256_of(dataset_path),
                    "rows": int(len(df)), "letters": len(stats)},
        "normalization": {key: [r.lo, r.hi] for key, r in frozen.items()},
        "poses": poses,
    }
    numbers = {"stats": stats, "overrides": overrides, "disagreements": disagreements,
               "recomputed": kinematics.dataset_ranges(landmarks, labels), "frozen": frozen}
    return doc, numbers


# ---------------------------------------------------------------- report

def _cell(stat, i=None) -> str:
    med, q1, q3 = (s if i is None else s[i] for s in stat)
    flag = " ⚠" if q3 - q1 > WIDE_IQR else ""
    return f"{med:.2f} [{q1:.2f}–{q3:.2f}]{flag}"


def unnatural(poses: Dict[str, Dict[str, Any]]) -> List[str]:
    """Every handshape violation across the final letter poses."""
    problems = []
    for letter in handshapes.HANDSHAPES:
        problems += handshapes.check(letter, HandPose.from_dict(poses[letter]))
    return problems


def gate_rows(poses: Dict[str, Dict[str, Any]], stats: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, bool]]:
    """ROADMAP stage 2 "done when" on the dataset medians (before shaping), plus the
    naturalness check on the final poses."""
    idx = {letter: float(stats[letter]["curl"][0][0]) for letter in stats}
    spread = {letter: float(stats[letter]["spread"][0]) for letter in stats}
    open_ok = all(idx[x] < 0.15 for x in "BWUV")
    fist_ok = all(idx[x] > 0.6 for x in "AST")
    diff = spread["V"] - spread["U"]
    names = set(poses)
    expected = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | {"REST", "OPEN"}
    problems = unnatural(poses)
    return [
        ("B/W/U/V index curl < 0.15 (dataset)", ", ".join(f"{x} {idx[x]:.3f}" for x in "BWUV"), open_ok),
        ("A/S/T index curl > 0.6 (dataset)", ", ".join(f"{x} {idx[x]:.3f}" for x in "AST"), fist_ok),
        ("V − U spread ≥ 0.3 (dataset)", f"{spread['V']:.3f} − {spread['U']:.3f} = {diff:.3f}", diff >= 0.3),
        ("poses.json has A–Z + REST + OPEN", f"{len(names & expected)}/{len(expected)}", expected <= names),
        ("every letter matches its ASL handshape", "all 26" if not problems else "; ".join(problems),
         not problems),
    ]


def _handshape_code(letter: str) -> str:
    spec = handshapes.HANDSHAPES[letter]
    code = "".join({"extended": "E", "closed": "C", "partial": "P"}[s] for s in spec.fingers)
    return f"{code} · thumb {spec.thumb} · {spec.spread}"


def render_report(doc: Dict[str, Any], numbers: Dict[str, Any]) -> str:
    stats, poses = numbers["stats"], doc["poses"]
    ds = doc["dataset"]
    lines = [
        "# Pose library report",
        "",
        f"Generated by `scripts/build_poses.py` on {doc['built']} from `{ds['path']}` "
        f"({ds['rows']:,} rows, {ds['letters']} letters, sha256 `{ds['sha256'][:12]}…`). "
        "Do not edit; rebuild instead. `python scripts/render_poses.py` draws every pose "
        "into `reports/poses_preview.png`.",
        "",
        "## Gate",
        "",
        "| Check | Value | Result |",
        "|---|---|---|",
    ]
    for check, value, ok in gate_rows(poses, stats):
        lines.append(f"| {check} | {value} | {'PASS' if ok else '**FAIL**'} |")

    lines += [
        "",
        "## Recordings that contradict the ASL handshape",
        "",
        "Measured curl (before shaping) that is the opposite state, not just MediaPipe's normal "
        "under-read of tucked fingers. The final pose is correct either way; re-recording these "
        "letters would make the data agree.",
        "",
    ]
    if numbers["disagreements"]:
        lines += ["| Letter | Handshape | Problem |", "|---|---|---|"]
        for letter, found in sorted(numbers["disagreements"].items()):
            lines.append(f"| {letter} | {_handshape_code(letter)} | {'; '.join(found)} |")
    else:
        lines.append("None.")

    lines += [
        "",
        "## Per-letter medians from the dataset (before shaping)",
        "",
        "Median over every frame of the letter, IQR in brackets; ⚠ marks an IQR wider than "
        f"{WIDE_IQR}. Curl: 0 = extended, 1 = full fist. Thumb opp: 0 = out (L), 1 = across the palm (M). "
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

    lines += ["", "## Overrides applied (`hand/pose_overrides.json`)", "",
              "`field <- X` copies letter X's shaped value.", ""]
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
        "Handshape: fingers index→pinky as E extended, C closed, P partial. *Shaped* lists the "
        "values the handshape set; *overridden* the ones `pose_overrides.json` set.",
        "",
        "| Pose | Index | Middle | Ring | Pinky | Thumb flex | Thumb opp | Spread | Roll | Handshape | Shaped | Overridden |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in sorted(poses, key=lambda n: (len(n) > 1, n)):
        p = poses[name]
        curls = " | ".join(f"{c:.2f}" for c in p["curl"])
        code = _handshape_code(name) if name in handshapes.HANDSHAPES else p["source"]
        shaped = ", ".join(f.replace("curl.", "") for f in p.get("shaped", [])) or "–"
        overridden = ", ".join(p.get("overridden", [])) or "–"
        lines.append(f"| {name} | {curls} | {p['thumb_flex']:.2f} | {p['thumb_opp']:.2f} | "
                     f"{p['spread']:.2f} | {p['wrist_roll']:+.2f} | {code} | {shaped} | {overridden} |")

    lines += [
        "",
        "## Normalization ranges",
        "",
        "Frozen in `hand/kinematics.py`; recomputed here from the dataset. Curl anchors are the median "
        "extended finger and the median A/S fist finger; the others are 2nd/98th percentiles. "
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
    failed = [(check, value) for check, value, ok in gate_rows(doc["poses"], numbers["stats"]) if not ok]
    for check, value in failed:
        print(f"GATE FAIL: {check}: {value}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
