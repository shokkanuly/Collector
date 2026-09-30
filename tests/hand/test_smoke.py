"""Stage 0 smoke tests: the hand package imports cleanly and stays optional."""
import os
import subprocess
import sys
import unittest

import yaml

from hand import JOINTS, N_CH, HandPose, ServoFrame

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HAND_MODULES = ("hand", "hand.types", "hand.kinematics", "hand.pose_library",
                "hand.mapping", "hand.protocol", "hand.link", "hand.controller")


class TestHandPackage(unittest.TestCase):

    def test_import_does_not_pull_in_serial(self):
        """CLAUDE.md rule 1: importing any hand module must never import pyserial."""
        # A fresh interpreter, so a test elsewhere that imports serial can't mask a regression.
        code = f"import sys\nfor m in {HAND_MODULES!r}: __import__(m)\nprint('serial' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False")

    def test_config_channels_match_joint_order(self):
        with open(os.path.join(REPO_ROOT, "config", "hand.yaml")) as f:
            cfg = yaml.safe_load(f)
        self.assertEqual([c["ch"] for c in cfg["channels"]], list(range(N_CH)))
        self.assertEqual(tuple(c["joint"] for c in cfg["channels"]), JOINTS)

    def test_value_types_validate(self):
        pose = HandPose(curl=(0.0, 0.5, 1.0, 0.2), thumb_flex=0.3, thumb_opp=0.9, spread=0.1)
        self.assertEqual(HandPose.from_dict(pose.to_dict()), pose)
        with self.assertRaises(ValueError):
            HandPose(curl=(0, 0, 0, 1.5), thumb_flex=0, thumb_opp=0, spread=0)
        self.assertEqual(ServoFrame((90,) * N_CH).angles_deg, (90,) * N_CH)
        with self.assertRaises(ValueError):
            ServoFrame((90,) * (N_CH - 1))


if __name__ == "__main__":
    unittest.main()
