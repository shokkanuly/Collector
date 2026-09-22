"""Stage 1: the PROTOCOL.md codec, including framing and malformed input."""
import unittest

from hand import N_CH
from hand.protocol import (
    MAX_LINE_BYTES, Angles, Err, ErrorCode, Home, LineAssembler, LineTooLong, Ok,
    Ping, Pong, ProtocolError, Query, Ready, Relax, SetAll, SetChannel, SetSlew,
    ShowLetter, Watchdog, decode_command, decode_reply, encode_command,
    encode_reply, is_blank, is_unsolicited,
)

# Every command in PROTOCOL.md §2 and §5, with its exact wire form.
COMMANDS = [
    (SetAll((90, 120, 0, 0, 0, 0, 90, 90)), b"S 90 120 0 0 0 0 90 90\n"),
    (SetChannel(3, 145), b"C 3 145\n"),
    (Home(), b"H\n"),
    (Relax(), b"X\n"),
    (Ping(), b"P\n"),
    (Query(), b"Q\n"),
    (SetSlew(6), b"V 6\n"),
    (ShowLetter("A"), b"L A\n"),
]

# Every reply in PROTOCOL.md §3.
REPLIES = [
    (Ok(), b"OK\n"),
    (Ok(clamped=True), b"OK C\n"),
    (Err(ErrorCode.UNKNOWN_COMMAND), b"ERR 1\n"),
    (Err(ErrorCode.LINE_TOO_LONG), b"ERR 2\n"),
    (Err(ErrorCode.BAD_VALUE), b"ERR 3\n"),
    (Err(ErrorCode.ESTOP), b"ERR 4\n"),
    (Pong(), b"PONG\n"),
    (Angles((90, 118, 3, 0, 0, 0, 90, 90)), b"A 90 118 3 0 0 0 90 90\n"),
    (Ready(version=1, channels=8), b"READY v1 ch=8\n"),
    (Watchdog(), b"WDT\n"),
]


class TestCommandCodec(unittest.TestCase):

    def test_encode_every_command(self):
        for cmd, wire in COMMANDS:
            with self.subTest(cmd=cmd):
                self.assertEqual(encode_command(cmd), wire)

    def test_round_trip_every_command(self):
        for cmd, wire in COMMANDS:
            with self.subTest(cmd=cmd):
                self.assertEqual(decode_command(wire), cmd)
                self.assertEqual(decode_command(encode_command(cmd)), cmd)

    def test_worst_case_s_line_fits(self):
        """PROTOCOL.md §4: S lines are at most 34 bytes."""
        line = encode_command(SetAll((180,) * N_CH))
        self.assertEqual(len(line), 34)
        self.assertLessEqual(len(line), MAX_LINE_BYTES)

    def test_encoder_rejects_values_python_must_never_send(self):
        bad = [
            SetAll((90,) * (N_CH - 1)),
            SetAll((90,) * (N_CH - 1) + (181,)),
            SetAll((90,) * (N_CH - 1) + (-1,)),
            SetChannel(N_CH, 90),
            SetChannel(-1, 90),
            SetChannel(0, 200),
            SetSlew(0),
            SetSlew(31),
            ShowLetter("a"),
            ShowLetter("AB"),
        ]
        for cmd in bad:
            with self.subTest(cmd=cmd), self.assertRaises(ValueError):
                encode_command(cmd)
        for cmd in (SetAll((90.0,) * N_CH), SetChannel(0, True), "P"):
            with self.subTest(cmd=cmd), self.assertRaises(TypeError):
                encode_command(cmd)

    def test_decoder_tolerates_cr_and_extra_spaces(self):
        self.assertEqual(decode_command(b"P\r\n"), Ping())
        self.assertEqual(decode_command(b"  C  3   145 \n"), SetChannel(3, 145))
        self.assertEqual(decode_command("V 007"), SetSlew(7))

    def test_decoder_keeps_out_of_range_angles_for_the_firmware_to_clamp(self):
        """PROTOCOL.md §2: angles are clamped, not rejected (answered OK C)."""
        self.assertEqual(decode_command(b"C 6 200\n"), SetChannel(6, 200))
        self.assertEqual(decode_command(b"S -5 0 0 0 0 0 0 9999\n"),
                         SetAll((-5, 0, 0, 0, 0, 0, 0, 9999)))
        self.assertEqual(decode_command(b"V 99\n"), SetSlew(99))

    def test_malformed_commands_carry_the_firmware_error_code(self):
        cases = [
            (b"Z\n", ErrorCode.UNKNOWN_COMMAND),              # unknown letter
            (b"p\n", ErrorCode.UNKNOWN_COMMAND),              # letters are upper case
            (b"PP\n", ErrorCode.UNKNOWN_COMMAND),
            (b"P 1\n", ErrorCode.UNKNOWN_COMMAND),            # too many fields
            (b"S 90 90 90\n", ErrorCode.UNKNOWN_COMMAND),     # too few fields
            (b"C 3\n", ErrorCode.UNKNOWN_COMMAND),
            (b"\xffP\n", ErrorCode.UNKNOWN_COMMAND),          # non-ASCII byte
            (b"\n", ErrorCode.UNKNOWN_COMMAND),               # blank (ignored by framing users)
            (b"C 3 abc\n", ErrorCode.BAD_VALUE),              # not an integer
            (b"C 3 14.5\n", ErrorCode.BAD_VALUE),
            (b"C 3 +5\n", ErrorCode.BAD_VALUE),
            (b"C 3 0x10\n", ErrorCode.BAD_VALUE),
            (b"C 3 12345\n", ErrorCode.BAD_VALUE),            # more than 4 digits
            (b"C 3 -\n", ErrorCode.BAD_VALUE),
            (b"C 3 \xd9\xa3\n", ErrorCode.BAD_VALUE),         # non-ASCII digit
            (b"C 8 90\n", ErrorCode.BAD_VALUE),               # channel out of range
            (b"C -1 90\n", ErrorCode.BAD_VALUE),
            (b"L a\n", ErrorCode.BAD_VALUE),
            (b"L 5\n", ErrorCode.BAD_VALUE),
        ]
        for line, code in cases:
            with self.subTest(line=line):
                with self.assertRaises(ProtocolError) as ctx:
                    decode_command(line)
                self.assertEqual(ctx.exception.code, code)

    def test_blank_lines(self):
        self.assertTrue(is_blank(b"\n"))
        self.assertTrue(is_blank(b"   \r\n"))
        self.assertFalse(is_blank(b"P\n"))


class TestReplyCodec(unittest.TestCase):

    def test_encode_every_reply(self):
        for reply, wire in REPLIES:
            with self.subTest(reply=reply):
                self.assertEqual(encode_reply(reply), wire)

    def test_round_trip_every_reply(self):
        for reply, wire in REPLIES:
            with self.subTest(reply=reply):
                self.assertEqual(decode_reply(wire), reply)
                self.assertEqual(decode_reply(encode_reply(reply)), reply)

    def test_only_ready_and_wdt_are_unsolicited(self):
        unsolicited = {type(r) for r, _ in REPLIES if is_unsolicited(r)}
        self.assertEqual(unsolicited, {Ready, Watchdog})

    def test_malformed_replies(self):
        for line in (b"", b"ok\n", b"OK X\n", b"ERR\n", b"ERR 9\n", b"ERR x\n",
                     b"A 1 2 3\n", b"A 1 2 3 4 5 6 7 x\n", b"READY v1\n",
                     b"READY vx ch=8\n", b"PONGPONG\n", b"\x00\n"):
            with self.subTest(line=line), self.assertRaises(ProtocolError):
                decode_reply(line)


class TestFraming(unittest.TestCase):

    def test_splits_lines_across_chunks(self):
        asm = LineAssembler()
        self.assertEqual(asm.feed(b"P\nC 3 1"), [b"P"])
        self.assertEqual(asm.feed(b"45\r\nQ"), [b"C 3 145\r"])
        self.assertEqual(asm.feed(b"\n"), [b"Q"])

    def test_limit_is_48_bytes_including_newline(self):
        asm = LineAssembler()
        exactly = b"S " + b"1" * (MAX_LINE_BYTES - 3) + b"\n"
        self.assertEqual(len(exactly), MAX_LINE_BYTES)
        self.assertEqual(asm.feed(exactly), [exactly[:-1]])
        one_over = b"S " + b"1" * (MAX_LINE_BYTES - 2) + b"\n"
        self.assertEqual(asm.feed(one_over), [LineTooLong()])

    def test_trailing_cr_counts_toward_the_limit(self):
        line = b"S " + b"1" * (MAX_LINE_BYTES - 3) + b"\r\n"
        self.assertEqual(LineAssembler().feed(line), [LineTooLong()])

    def test_overlong_line_is_dropped_once_and_framing_recovers(self):
        asm = LineAssembler()
        events = asm.feed(b"X" * 200 + b"\nP\n")
        self.assertEqual(events, [LineTooLong(), b"P"])


if __name__ == "__main__":
    unittest.main()
