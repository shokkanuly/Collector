"""Stage 2: kinematics on synthetic hands with known geometry, then on the real dataset."""
import os
import unittest

import numpy as np
import pandas as pd

from hand import HandPose
from hand import kinematics as k
from hand.types import LANDMARK_COLUMNS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = os.path.join(REPO_ROOT, "landmarks_dataset.csv")

MCPS = {"index": (-0.3, -1.0), "middle": (-0.1, -1.0), "ring": (0.1, -0.95), "pinky": (0.3, -0.85)}
FINGER_IDS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
PALM_NORMAL = np.array([0.0, 0.0, 1.0])


def rotate(v, axis, deg):
    """Rodrigues rotation of v about a unit axis."""
    t = np.radians(deg)
    return v * np.cos(t) + np.cross(axis, v) * np.sin(t) + axis * np.dot(axis, v) * (1 - np.cos(t))


def synthetic_hand(flex_deg=(0, 0, 0, 0), thumb_bend_deg=0.0, thumb_tip=None, index_abduct_deg=0.0,
                   roll_deg=0.0):
    """A hand pointing up the image with its palm toward the camera, as 63 floats.

    Each finger continues its wrist -> MCP line and bends toward the palm by
    flex_deg at MCP, PIP, and DIP, so curl_deg is exactly 3 * flex_deg. The
    index can be abducted away from the middle finger in the palm plane, and
    the whole hand rolled about the image's vertical axis.
    """
    p = np.zeros((21, 3))
    for (finger, (x, y)), flex in zip(MCPS.items(), flex_deg):
        mcp, pip, dip, tip = FINGER_IDS[finger]
        p[mcp] = (x, y, 0.0)
        d = p[mcp] / np.linalg.norm(p[mcp])
        if finger == "index":
            d = rotate(d, PALM_NORMAL, -index_abduct_deg)  # away from the middle finger (-x)
        axis = np.cross(d, PALM_NORMAL)
        axis /= np.linalg.norm(axis)
        prev = p[mcp]
        for joint, length in zip((pip, dip, tip), (0.35, 0.25, 0.2)):
            d = rotate(d, axis, flex)
            p[joint] = prev + length * d
            prev = p[joint]
    # Thumb: CMC -> MCP -> IP -> tip, pointing out to the side, bending in the palm plane.
    p[1] = (-0.35, -0.25, 0.0)
    d = np.array([-0.8, -0.6, 0.0])
    for joint, length in zip((2, 3, 4), (0.3, 0.25, 0.2)):
        p[joint] = p[joint - 1] + length * d
        d = rotate(d, PALM_NORMAL, -thumb_bend_deg)
    if thumb_tip is not None:
        p[4] = thumb_tip
    c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    roll = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # about the image y axis
    return (p @ roll.T).ravel()


class TestSyntheticGeometry(unittest.TestCase):

    def test_straight_hand_has_zero_curl(self):
        np.testing.assert_allclose(k.curl_deg(synthetic_hand()), 0.0, atol=1e-9)
        np.testing.assert_allclose(k.curl_01(synthetic_hand()), 0.0)

    def test_curl_is_the_sum_of_three_bends(self):
        for flex in (10, 30, 60, 90):
            with self.subTest(flex=flex):
                deg = k.curl_deg(synthetic_hand(flex_deg=(flex, 0, flex, 0)))
                np.testing.assert_allclose(deg, [3 * flex, 0, 3 * flex, 0], atol=1e-9)

    def test_every_finger_shares_one_curl_range(self):
        """The same bend reads the same on every finger, so partial shapes (C, O) stay even."""
        half = (k.CURL_RANGE_DEG.lo + k.CURL_RANGE_DEG.hi) / 2 / 3  # per joint, halfway through the range
        np.testing.assert_allclose(k.curl_01(synthetic_hand(flex_deg=(half,) * 4)), 0.5, atol=1e-9)

    def test_thumb_flex_sums_the_two_thumb_bends(self):
        self.assertAlmostEqual(k.thumb_flex_deg(synthetic_hand()), 0.0, places=6)
        self.assertAlmostEqual(k.thumb_flex_deg(synthetic_hand(thumb_bend_deg=25)), 50.0, places=6)

    def test_thumb_across_the_palm_reads_more_opposed_than_out(self):
        out = k.thumb_opp_01(synthetic_hand())  # a short synthetic thumb: not as far out as L
        across = k.thumb_opp_01(synthetic_hand(thumb_tip=(0.25, -0.75, 0.1)))
        self.assertLess(out, 0.5)
        self.assertGreater(across, 0.9)

    def test_spread_tracks_abduction(self):
        base = k.spread_deg(synthetic_hand())
        wide = k.spread_deg(synthetic_hand(index_abduct_deg=10))
        self.assertAlmostEqual(wide - base, 10.0, places=6)
        self.assertGreater(k.spread_01(synthetic_hand(index_abduct_deg=10)), k.spread_01(synthetic_hand()))

    def test_spread_fades_to_zero_when_a_finger_curls(self):
        wide = dict(index_abduct_deg=10)
        self.assertGreater(k.spread_01(synthetic_hand(**wide)), 0.5)
        curled = k.spread_01(synthetic_hand(flex_deg=(0, 70, 0, 0), **wide))  # middle curl_01 ~0.86
        self.assertEqual(curled, 0.0)

    def test_wrist_roll_follows_the_palm(self):
        for roll in (-80, -45, 0, 30, 60):
            with self.subTest(roll=roll):
                self.assertAlmostEqual(k.wrist_roll_deg(synthetic_hand(roll_deg=roll)), roll, places=6)
                self.assertAlmostEqual(k.wrist_roll(synthetic_hand(roll_deg=roll)), roll / 90.0, places=6)

    def test_shape_measurements_ignore_hand_orientation(self):
        hand = synthetic_hand(flex_deg=(20, 40, 60, 30), thumb_bend_deg=15, index_abduct_deg=5)
        rng = np.random.default_rng(0)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        turned = (hand.reshape(21, 3) @ q.T).ravel()
        for fn in (k.curl_deg, k.thumb_flex_deg, k.thumb_pinky_dist, k.spread_deg):
            with self.subTest(fn=fn.__name__):
                np.testing.assert_allclose(fn(turned), fn(hand), atol=1e-9)

    def test_batch_matches_single_hands(self):
        hands = [synthetic_hand(flex_deg=(f, 2 * f, 0, f), thumb_bend_deg=f, roll_deg=f - 20)
                 for f in (0, 10, 25, 40)]
        batch = np.stack(hands)
        features = k.pose_features(batch)
        for i, hand in enumerate(hands):
            pose = k.pose_from_landmarks(hand)
            np.testing.assert_allclose(features["curl"][i], pose.curl)
            for field in ("thumb_flex", "thumb_opp", "spread", "wrist_roll"):
                self.assertAlmostEqual(features[field][i], getattr(pose, field))
        self.assertEqual(k.curl_01(batch).shape, (4, 4))
        self.assertEqual(k.spread_01(batch).shape, (4,))
        self.assertIsInstance(k.spread_01(hands[0]), float)

    def test_degenerate_hand_still_gives_a_valid_pose(self):
        self.assertIsInstance(k.pose_from_landmarks(np.zeros(63)), HandPose)

    def test_bad_input_is_rejected(self):
        for bad in (np.zeros(62), np.zeros((2, 3, 63))):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                k.curl_deg(bad)
        with self.assertRaises(ValueError):
            k.pose_from_landmarks(np.zeros((2, 63)))
        nan = np.zeros(63)
        nan[7] = np.nan
        with self.assertRaises(ValueError):
            k.pose_from_landmarks(nan)


class TestKinematicsOnDataset(unittest.TestCase):
    """ARCHITECTURE §8 and the ROADMAP stage-2 gate, on landmarks_dataset.csv (read-only)."""

    @classmethod
    def setUpClass(cls):
        df = pd.read_csv(DATASET)
        cls.X = df[list(LANDMARK_COLUMNS)].to_numpy()
        cls.labels = df["label"].to_numpy()
        cls.sessions = df["session_id"].to_numpy()
        cls.f = k.pose_features(cls.X)

    def median(self, field, letter, finger=None):
        values = self.f[field][self.labels == letter]
        if finger is not None:
            values = values[:, ("index", "middle", "ring", "pinky").index(finger)]
        return float(np.median(values))

    def test_open_fingers_read_straight(self):
        for letter in "BWUV":
            with self.subTest(letter=letter):
                self.assertLess(self.median("curl", letter, "index"), 0.15)
        for letter in "BW":
            with self.subTest(letter=letter):
                self.assertLess(self.median("curl", letter, "middle"), 0.15)

    def test_fists_read_curled(self):
        for letter in "AS":
            for finger in ("index", "middle", "ring", "pinky"):
                with self.subTest(letter=letter, finger=finger):
                    self.assertGreater(self.median("curl", letter, finger), 0.6)
        self.assertGreater(self.median("curl", "T", "index"), 0.6)

    def test_pinky_up_letters(self):
        for letter in "IY":
            with self.subTest(letter=letter):
                self.assertLess(self.median("curl", letter, "pinky"), 0.15)

    def test_thumb_out_reads_less_opposed_than_tucked(self):
        self.assertLess(self.median("thumb_opp", "L"), self.median("thumb_opp", "U"))

    def test_spread_separates_u_from_v(self):
        self.assertGreaterEqual(self.median("spread", "V") - self.median("spread", "U"), 0.3)
        # Session by session, not just in the pooled medians.
        per_session = pd.Series(self.f["spread"]).groupby(self.sessions).median()
        u = per_session[[s for s in per_session.index if s.startswith("U_")]]
        v = per_session[[s for s in per_session.index if s.startswith("V_")]]
        self.assertLess(u.max(), v.min())

    def test_frozen_ranges_still_match_the_dataset(self):
        """A dataset change that moves a percentile shows up here, not as silent drift."""
        recomputed = k.dataset_ranges(self.X, self.labels)
        for key, frozen in k.frozen_ranges().items():
            width = abs(frozen.hi - frozen.lo)
            with self.subTest(range=key):
                self.assertAlmostEqual(recomputed[key].lo, frozen.lo, delta=0.02 * width)
                self.assertAlmostEqual(recomputed[key].hi, frozen.hi, delta=0.02 * width)


if __name__ == "__main__":
    unittest.main()
