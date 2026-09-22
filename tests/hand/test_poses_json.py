"""Stage 2: hand/poses.json (ARCHITECTURE §8) and the PoseLibrary that reads it."""
import json
import unittest

from hand import HandPose
from hand.pose_library import DEFAULT_PATH, PoseLibrary
from scripts.build_poses import DATASET, OVERRIDES, build, load_overrides, sha256_of

ALL_NAMES = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") | {"REST", "OPEN"}
REBUILD = "run: python scripts/build_poses.py"


class TestPosesJson(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(DEFAULT_PATH) as f:
            cls.doc = json.load(f)

    def test_covers_all_26_letters_plus_rest_and_open(self):
        self.assertEqual(set(self.doc["poses"]), ALL_NAMES)

    def test_dataset_hash_matches_the_csv(self):
        self.assertEqual(self.doc["dataset"]["sha256"], sha256_of(DATASET),
                         f"landmarks_dataset.csv changed since poses.json was built; {REBUILD}")

    def test_matches_a_fresh_build(self):
        """Also catches a stale file after the kinematics or the overrides change."""
        fresh, _ = build(built=self.doc["built"])
        self.assertEqual(json.loads(json.dumps(fresh)), self.doc, f"hand/poses.json is stale; {REBUILD}")

    def test_every_pose_is_a_valid_handpose(self):
        for name, entry in self.doc["poses"].items():
            with self.subTest(pose=name):
                HandPose.from_dict(entry)

    def test_overrides_are_explained_and_recorded(self):
        for letter, override in load_overrides(OVERRIDES).items():
            with self.subTest(letter=letter):
                self.assertTrue(override.get("reason", "").strip())
                entry = self.doc["poses"][letter]
                self.assertTrue(entry["overridden"])
                self.assertEqual(entry["reason"], override["reason"].strip())

    def test_motion_letters_are_stubs_with_a_note(self):
        for letter in "JZ":
            with self.subTest(letter=letter):
                self.assertEqual(self.doc["poses"][letter]["source"], "motion-stub")
                self.assertIn("motion", self.doc["poses"][letter])


class TestPoseLibrary(unittest.TestCase):

    def setUp(self):
        self.lib = PoseLibrary.load()

    def test_has_every_letter(self):
        self.assertEqual(self.lib.letters, tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        self.assertEqual(len(self.lib), len(ALL_NAMES))

    def test_lookup_ignores_case(self):
        self.assertEqual(self.lib.pose_for("a"), self.lib.pose_for("A"))
        self.assertIn("rest", self.lib)
        self.assertNotIn("?", self.lib)
        self.assertNotIn(5, self.lib)

    def test_named_poses(self):
        rest, open_hand = self.lib.rest, self.lib.open
        # REST is a relaxed hand: slightly bent, more toward the pinky, thumb relaxed.
        self.assertEqual(list(rest.curl), sorted(rest.curl))
        self.assertTrue(all(0.05 <= c <= 0.3 for c in rest.curl))
        self.assertLessEqual(rest.thumb_flex, 0.3)
        self.assertEqual(rest.wrist_roll, 0.0)
        # OPEN is the flat, spread "5" hand.
        self.assertEqual(open_hand, HandPose(curl=(0, 0, 0, 0), thumb_flex=0, thumb_opp=0, spread=1))

    def test_unknown_names_raise(self):
        for name in ("?", "1", "AB", ""):
            with self.subTest(name=name), self.assertRaises(KeyError):
                self.lib.pose_for(name)

    def test_motion_letters_fall_back_to_their_handshape_and_log_once(self):
        with self.assertLogs("hand.pose_library", level="INFO") as logs:
            j = self.lib.pose_for("J")
            self.lib.pose_for("j")
            self.lib.pose_for("Z")
        self.assertEqual(len(logs.records), 2)
        self.assertEqual(j, self.lib.pose_for("I"))

    def test_info_says_where_a_pose_came_from(self):
        self.assertEqual(self.lib.info("J")["source"], "motion-stub")
        self.assertEqual(self.lib.info("REST")["source"], "named")
        self.assertNotIn("curl", self.lib.info("A"))

    def test_records_the_dataset_hash(self):
        self.assertEqual(self.lib.dataset_sha256, sha256_of(DATASET))


if __name__ == "__main__":
    unittest.main()
