"""Draw every pose in hand/poses.json as a hand skeleton: reports/poses_preview.png.

    python scripts/render_poses.py [--poses hand/poses.json] [--out reports/poses_preview.png]

A quick way to see whether poses look natural before any hardware exists. Each
letter gets a front view (as the viewer sees the palm) and a side view, where
curl is easiest to judge: closed fingers wrap back toward the palm, extended
ones stand straight.

This is an approximate forward-kinematics preview, not the real mechanism. It
splits a finger's curl over MCP, PIP, and DIP in the human-like ratio of the
Blender model's fist (88:100:63 deg at curl 1), which is also how a single
tendon closes a finger. The thumb follows a guessed path: out (L), beside the
index (A), across the palm (B, M). The real joint coupling and thumb path are
set by the printed hand and its calibration (ROADMAP stage 4).
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSES_JSON = os.path.join(REPO_ROOT, "hand", "poses.json")
PREVIEW = os.path.join(REPO_ROOT, "reports", "poses_preview.png")

# x = viewer's right, y = up, z = toward the viewer; the palm faces +z.
# Unit: wrist to middle knuckle ~ 1 (~9 cm on a human hand).
FINGERS = ("index", "middle", "ring", "pinky")
MCP_XY = {"index": (0.30, 0.98), "middle": (0.10, 1.03), "ring": (-0.10, 0.98), "pinky": (-0.28, 0.88)}
PHALANGES = {"index": (0.44, 0.27, 0.20), "middle": (0.48, 0.30, 0.21),
             "ring": (0.45, 0.28, 0.20), "pinky": (0.35, 0.22, 0.18)}
FIST_JOINT_DEG = (88.0, 100.0, 63.0)                 # MCP, PIP, DIP at curl = 1
SPREAD_DEG = {"index": 18.0, "middle": -8.0}         # abduction at spread = 1
SPLAY_DEG = {"ring": -3.0, "pinky": -8.0}            # fixed natural splay
THUMB_CMC = np.array([0.30, 0.22, 0.02])
THUMB_BONES = (0.36, 0.28, 0.21)                     # metacarpal, proximal, distal
THUMB_FLEX_DEG = (0.0, 55.0, 80.0)                   # bends at CMC, MCP, IP at flex = 1
THUMB_OUT = np.array([0.85, 0.5, 0.15])              # thumb_opp 0
THUMB_BESIDE = np.array([0.25, 1.0, 0.35])           # thumb_opp 0.5
THUMB_ACROSS = np.array([-0.75, 0.45, 0.55])         # thumb_opp 1
COLORS = ("#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#7f3fbf")  # index..pinky, thumb


def _rotation(axis: Sequence[float], deg: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = np.radians(deg)
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k


def _slerp(u: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    w = np.arccos(np.clip(u @ v, -1.0, 1.0))
    return u if w < 1e-6 else (np.sin((1 - t) * w) * u + np.sin(t * w) * v) / np.sin(w)


def skeleton(pose: Dict) -> List[np.ndarray]:
    """Joint chains (wrist first) for index, middle, ring, pinky, and thumb."""
    chains = []
    for i, finger in enumerate(FINGERS):
        abduct = SPREAD_DEG.get(finger, 0.0) * pose["spread"] + SPLAY_DEG.get(finger, 0.0)
        direction = _rotation((0, 0, 1), -abduct) @ np.array([0.0, 1.0, 0.0])
        flex_axis = np.cross(direction, (0, 0, 1))
        point = np.array([*MCP_XY[finger], 0.0])
        chain = [np.zeros(3), point]
        for joint_deg, length in zip(FIST_JOINT_DEG, PHALANGES[finger]):
            direction = _rotation(flex_axis, -joint_deg * pose["curl"][i]) @ direction
            point = point + length * direction
            chain.append(point)
        chains.append(np.array(chain))

    opp = pose["thumb_opp"]
    direction = (_slerp(THUMB_OUT, THUMB_BESIDE, opp / 0.5) if opp <= 0.5
                 else _slerp(THUMB_BESIDE, THUMB_ACROSS, (opp - 0.5) / 0.5))
    flex_axis = np.cross(direction, (0, 0, 1))
    point = THUMB_CMC
    chain = [np.zeros(3), point]
    for bend, length in zip(THUMB_FLEX_DEG, THUMB_BONES):
        direction = _rotation(flex_axis, -bend * pose["thumb_flex"]) @ direction
        point = point + length * direction
        chain.append(point)
    chains.append(np.array(chain))

    roll = _rotation((0, 1, 0), 90.0 * pose["wrist_roll"])  # forearm axis is vertical
    return [chain @ roll.T for chain in chains]


def render(poses: Dict[str, Dict], names: Sequence[str], path: str, title: str, cols: int = 7) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2.6, rows * 2.2),
                             gridspec_kw={"width_ratios": [1.25, 1.0] * cols})
    for ax in axes.ravel():
        ax.set_axis_off()
    for k, name in enumerate(names):
        row, col = divmod(k, cols)
        front, side = axes[row, 2 * col], axes[row, 2 * col + 1]
        chains = skeleton(poses[name])
        palm = np.array([c[1] for c in chains[:4]] + [np.zeros(3), chains[4][1]])
        for ax, (h, v) in ((front, (0, 1)), (side, (2, 1))):
            ax.fill(palm[:, h], palm[:, v], color="#eee", zorder=0)
            # Front view: draw far chains first so near ones stay visible.
            order = sorted(range(5), key=lambda i: chains[i][:, 2].mean()) if h == 0 else range(5)
            for i in order:
                ax.plot(chains[i][:, h], chains[i][:, v], color=COLORS[i], lw=2.4, solid_capstyle="round")
                ax.plot(chains[i][-1, h], chains[i][-1, v], "o", color=COLORS[i], ms=3)
            ax.set_aspect("equal")
            ax.set_ylim(-0.1, 2.05)
            ax.set_xlim((-0.75, 1.0) if h == 0 else (-0.35, 1.1))
        front.set_title(name, fontsize=12, loc="right", pad=2)
        side.set_title("side", fontsize=7, color="#888", pad=2)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=72)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--poses", default=POSES_JSON)
    parser.add_argument("--out", default=PREVIEW)
    args = parser.parse_args(argv)
    with open(args.poses) as f:
        poses = json.load(f)["poses"]
    names = sorted(poses, key=lambda n: (len(n) > 1, n))
    render(poses, names, args.out, "hand/poses.json — front and side views. Index red, middle orange, "
           "ring green, pinky blue, thumb purple. Side view: fingers curl to the right, toward the viewer.")
    print(f"Wrote {os.path.relpath(args.out, REPO_ROOT)} ({len(names)} poses)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
