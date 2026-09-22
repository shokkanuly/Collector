"""Stage 2: HandPose -> ServoFrame mapping (ARCHITECTURE §4.3, §8)."""
import random
import unittest

from hand import N_CH, HandPose, ServoFrame
from hand.mapping import ChannelConfig, HandConfig, ServoMapper, config_from_dict, load_config
from hand.pose_library import PoseLibrary
from hand.protocol import MAX_LINE_BYTES, SetAll, encode_command
from hand.types import JOINTS

REST = HandPose(curl=(0, 0, 0, 0), thumb_flex=0, thumb_opp=0, spread=0, wrist_roll=0)
FIST = HandPose(curl=(1, 1, 1, 1), thumb_flex=1, thumb_opp=1, spread=1, wrist_roll=1)


def channels(**overrides):
    """A plain 0-180 config, one channel per joint, with per-joint tweaks."""
    return [ChannelConfig(ch=i, joint=j, **{"min_deg": 0, "max_deg": 180, **overrides.get(j, {})})
            for i, j in enumerate(JOINTS)]


def random_pose(rng):
    return HandPose(curl=tuple(rng.random() for _ in range(4)), thumb_flex=rng.random(),
                    thumb_opp=rng.random(), spread=rng.random(), wrist_roll=rng.uniform(-1, 1))


class TestRepoConfig(unittest.TestCase):
    """config/hand.yaml as committed (ARCHITECTURE §4.3 defaults, before calibration)."""

    def setUp(self):
        self.mapper = ServoMapper.from_config(load_config())

    def test_rest_opens_every_finger_whichever_way_its_servo_turns(self):
        # thumb_flex 10, thumb_opp 20, index 0, middle 180 (inverted), ring 0,
        # pinky 180 (inverted), spread 60, wrist 90 (roll 0 = mid-range).
        self.assertEqual(self.mapper.to_frame(REST).angles_deg, (10, 20, 0, 180, 0, 180, 60, 90))

    def test_fist_closes_every_finger(self):
        self.assertEqual(self.mapper.to_frame(FIST).angles_deg, (170, 160, 180, 0, 180, 0, 120, 180))

    def test_every_library_pose_maps_to_an_encodable_frame(self):
        library = PoseLibrary.load()
        for name in library.names:
            with self.subTest(pose=name):
                frame = self.mapper.to_frame(library.pose_for(name))
                self.assertLessEqual(len(encode_command(SetAll(frame.angles_deg))), MAX_LINE_BYTES)

    def test_config_values(self):
        cfg = load_config()
        self.assertEqual((cfg.port, cfg.baud, cfg.rate_hz), ("auto", 115200, 30.0))
        self.assertEqual((cfg.letter_hold_ms, cfg.smoothing_alpha), (900, 0.35))


class TestMappingRules(unittest.TestCase):

    def test_linear_between_min_and_max(self):
        mapper = ServoMapper(channels(index={"min_deg": 20, "max_deg": 120}))
        ch = mapper.channels[JOINTS.index("index")]
        self.assertEqual([mapper.angle_deg(ch, v) for v in (0, 0.25, 0.5, 1)], [20, 45, 70, 120])

    def test_inversion_mirrors_the_range(self):
        plain = ChannelConfig(ch=2, joint="index", min_deg=30, max_deg=150)
        inverted = ChannelConfig(ch=2, joint="index", min_deg=30, max_deg=150, invert=True)
        rng = random.Random(1)
        for _ in range(200):
            v = rng.random()
            self.assertEqual(ServoMapper.angle_deg(inverted, v), ServoMapper.angle_deg(plain, 1 - v))
        self.assertEqual(ServoMapper.angle_deg(inverted, 0.0), 150)
        self.assertEqual(ServoMapper.angle_deg(inverted, 1.0), 30)

    def test_min_equal_max_pins_the_channel(self):
        mapper = ServoMapper(channels(spread={"min_deg": 90, "max_deg": 90}, wrist_roll={"min_deg": 90, "max_deg": 90}))
        for pose in (REST, FIST):
            frame = mapper.to_frame(pose)
            self.assertEqual(frame.angles_deg[JOINTS.index("spread")], 90)
            self.assertEqual(frame.angles_deg[JOINTS.index("wrist_roll")], 90)

    def test_wrist_roll_spans_the_whole_range(self):
        mapper = ServoMapper(channels(wrist_roll={"min_deg": 40, "max_deg": 140}))
        wrist = JOINTS.index("wrist_roll")
        for roll, angle in ((-1, 40), (0, 90), (1, 140)):
            pose = HandPose(curl=(0, 0, 0, 0), thumb_flex=0, thumb_opp=0, spread=0, wrist_roll=roll)
            self.assertEqual(mapper.to_frame(pose).angles_deg[wrist], angle)

    def test_rounds_half_up(self):
        ch = ChannelConfig(ch=2, joint="index", min_deg=0, max_deg=1)
        self.assertEqual(ServoMapper.angle_deg(ch, 0.5), 1)
        self.assertEqual(ServoMapper.angle_deg(ch, 0.49), 0)

    def test_out_of_range_values_are_clamped(self):
        ch = ChannelConfig(ch=2, joint="index", min_deg=10, max_deg=170)
        self.assertEqual(ServoMapper.angle_deg(ch, -0.5), 10)
        self.assertEqual(ServoMapper.angle_deg(ch, 1.5), 170)

    def test_random_poses_stay_inside_every_channel_range(self):
        cfg = channels(thumb_flex={"min_deg": 10, "max_deg": 170},
                       middle={"min_deg": 5, "max_deg": 175, "invert": True},
                       spread={"min_deg": 60, "max_deg": 120})
        mapper = ServoMapper(cfg)
        rng = random.Random(7)
        for _ in range(1000):
            frame = mapper.to_frame(random_pose(rng))
            self.assertIsInstance(frame, ServoFrame)
            for c, angle in zip(mapper.channels, frame.angles_deg):
                self.assertTrue(c.min_deg <= angle <= c.max_deg, (c, angle))

    def test_monotonic_in_each_value(self):
        plain = ChannelConfig(ch=2, joint="index", min_deg=0, max_deg=180)
        inverted = ChannelConfig(ch=3, joint="middle", min_deg=0, max_deg=180, invert=True)
        values = [i / 100 for i in range(101)]
        up = [ServoMapper.angle_deg(plain, v) for v in values]
        down = [ServoMapper.angle_deg(inverted, v) for v in values]
        self.assertEqual(up, sorted(up))
        self.assertEqual(down, sorted(down, reverse=True))


class TestConfigValidation(unittest.TestCase):

    def test_bad_channels_are_rejected(self):
        bad = [
            dict(ch=0, joint="elbow", min_deg=0, max_deg=180),
            dict(ch=8, joint="index", min_deg=0, max_deg=180),
            dict(ch=0, joint="index", min_deg=100, max_deg=50),
            dict(ch=0, joint="index", min_deg=-5, max_deg=50),
            dict(ch=0, joint="index", min_deg=0, max_deg=181),
            dict(ch=0, joint="index", min_deg=0.5, max_deg=90),
            dict(ch=0, joint="index", min_deg=0, max_deg=90, invert="yes"),
        ]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ChannelConfig(**kwargs)

    def test_mapper_needs_each_channel_and_joint_once(self):
        good = channels()
        with self.assertRaises(ValueError):
            ServoMapper(good[:-1])
        duplicate_joint = good[:-1] + [ChannelConfig(ch=N_CH - 1, joint="index", min_deg=0, max_deg=180)]
        with self.assertRaises(ValueError):
            ServoMapper(duplicate_joint)
        ServoMapper(list(reversed(good)))  # order in the file does not matter

    def test_config_fields_are_checked(self):
        raw = {"channels": [dict(ch=c.ch, joint=c.joint, min_deg=c.min_deg, max_deg=c.max_deg)
                            for c in channels()]}
        self.assertIsInstance(config_from_dict(raw), HandConfig)
        for key, value in (("smoothing_alpha", 0), ("smoothing_alpha", 1.5), ("rate_hz", 0),
                           ("letter_hold_ms", -1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                config_from_dict(dict(raw, **{key: value}))


if __name__ == "__main__":
    unittest.main()
