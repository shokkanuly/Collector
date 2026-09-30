"""Naturalness: every pose matches its ASL handshape (hand/handshapes.py)."""
import json
import unittest

from hand import HandPose
from hand import handshapes as hs
from hand.pose_library import DEFAULT_PATH
from scripts.build_poses import build

KNOWN_BAD_RECORDINGS = {"F", "G", "L"}  # ARCHITECTURE §9: recollect these


class TestHandshapeTable(unittest.TestCase):

    def test_covers_the_whole_alphabet(self):
        self.assertEqual(set(hs.HANDSHAPES), set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        for letter, shape in hs.HANDSHAPES.items():
            with self.subTest(letter=letter):
                self.assertEqual(len(shape.fingers), 4)
                self.assertIn(shape.thumb, hs.THUMB_BOUNDS)
                self.assertIn(shape.spread, hs.SPREAD_TARGET)

    def test_shaping_sets_definitive_states_and_keeps_partial_ones(self):
        pose = {"curl": [0.1, 0.62, 0.58, 0.61], "thumb_flex": 0.5, "thumb_opp": 0.9, "spread": 0.3}
        changed = hs.shape_pose("U", pose)  # E E C C, together
        self.assertEqual(pose["curl"], [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(pose["spread"], 0.0)
        self.assertEqual(changed, ["curl.index", "curl.middle", "curl.ring", "curl.pinky", "spread"])
        c_pose = {"curl": [0.63, 0.66, 0.66, 0.60], "thumb_flex": 0.5, "thumb_opp": 0.8, "spread": 0.0}
        self.assertEqual(hs.shape_pose("C", c_pose), [])  # all partial: measured curves stay
        self.assertEqual(c_pose["curl"], [0.63, 0.66, 0.66, 0.60])

    def test_check_catches_half_closed_fingers_and_a_wrong_thumb(self):
        half_bent_u = HandPose(curl=(0, 0, 0.58, 0.61), thumb_flex=0.5, thumb_opp=0.94, spread=0)
        problems = hs.check("U", half_bent_u)
        self.assertEqual(len(problems), 2)
        self.assertIn("ring should be closed", problems[0])
        tucked_y_thumb = HandPose(curl=(1, 1, 1, 0), thumb_flex=0.03, thumb_opp=0.63, spread=0)
        self.assertTrue(any("thumb should be out" in p for p in hs.check("Y", tucked_y_thumb)))


class TestPosesAreNatural(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(DEFAULT_PATH) as f:
            cls.poses = json.load(f)["poses"]

    def test_every_letter_matches_its_handshape(self):
        for letter in hs.HANDSHAPES:
            with self.subTest(letter=letter):
                self.assertEqual(hs.check(letter, HandPose.from_dict(self.poses[letter])), [])

    def test_closed_fingers_close_fully_and_extended_ones_straighten(self):
        """The robot must not stop a tucked finger half-way: it would stick out from the palm."""
        for letter, shape in hs.HANDSHAPES.items():
            for finger, state, value in zip(("index", "middle", "ring", "pinky"), shape.fingers,
                                            self.poses[letter]["curl"]):
                if state in hs.CURL_TARGET:
                    with self.subTest(letter=letter, finger=finger):
                        self.assertEqual(value, hs.CURL_TARGET[state])

    def test_partial_shapes_are_even(self):
        """C, O, and E curve every finger alike; lopsided curls look unnatural."""
        for letter in "COE":
            curl = self.poses[letter]["curl"]
            with self.subTest(letter=letter):
                self.assertLess(max(curl) - min(curl), 0.15)

    def test_no_new_recording_contradicts_asl(self):
        """A newly recorded letter that contradicts its handshape must be noticed, not shaped away."""
        _, numbers = build(built="test")
        self.assertLessEqual(set(numbers["disagreements"]), KNOWN_BAD_RECORDINGS)


class TestPreview(unittest.TestCase):

    def test_preview_renders_every_pose(self):
        import os
        import tempfile

        import numpy as np

        from scripts.render_poses import main, skeleton

        with open(DEFAULT_PATH) as f:
            poses = json.load(f)["poses"]
        for name, pose in poses.items():
            with self.subTest(pose=name):
                chains = skeleton(pose)
                self.assertEqual([len(c) for c in chains], [5, 5, 5, 5, 5])
                self.assertTrue(all(np.all(np.isfinite(c)) for c in chains))
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "preview.png")
            self.assertEqual(main(["--out", out]), 0)
            self.assertGreater(os.path.getsize(out), 10_000)


if __name__ == "__main__":
    unittest.main()
