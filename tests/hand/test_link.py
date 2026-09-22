"""Stage 1: the firmware model, MockLink, and SerialLink (through a fake port)."""
import os
import subprocess
import sys
import unittest

from hand import N_CH, ServoFrame
from hand.link import (
    POLL_S, FirmwareSim, HandshakeError, LinkDisconnected, LinkTimeout, MockLink,
    SerialLink, find_port,
)
from hand.protocol import (
    ACK_TIMEOUT_S, KEEPALIVE_S, READY_TIMEOUT_S, Err, ErrorCode, Home, Ok, Ping,
    Query, Ready, Relax, SetAll, SetChannel, SetSlew, ShowLetter, Watchdog,
)
from tests.hand.fakes import FakeClock, SimSerialPort

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REST = (90,) * N_CH
LETTER_A = (25, 140, 150, 170, 170, 150, 90, 90)
EPS = 1e-9  # FakeClock starts at 1000 s, so sums of 5 ms steps carry float error
SPREAD_LIMITS = dict(hard_min_deg=(0,) * 6 + (60, 0), hard_max_deg=(180,) * 6 + (120, 180))


def drain(sim):
    """All complete lines the model has printed so far."""
    out = []
    line = sim.read_line()
    while line is not None:
        out.append(line)
        line = sim.read_line()
    return out


def booted_sim(**kwargs):
    clock = FakeClock()
    sim = FirmwareSim(clock=clock, **kwargs)
    drain(sim)  # the READY from power-on
    return sim, clock


class TestFirmwareSim(unittest.TestCase):
    """The executable spec the stage-3 sketch is bench-tested against."""

    def test_boot_announces_ready_with_servos_limp(self):
        sim = FirmwareSim(clock=FakeClock())
        self.assertEqual(drain(sim), [b"READY v1 ch=8\n"])
        self.assertFalse(sim.pwm_on)

    def test_error_replies_for_raw_lines(self):
        sim, _ = booted_sim()
        cases = [
            (b"Z\n", b"ERR 1\n"),
            (b"S 1 2 3\n", b"ERR 1\n"),
            (b"L A\n", b"ERR 1\n"),              # a normal build has no L command
            (b"X" * 60 + b"\n", b"ERR 2\n"),
            (b"C 3 abc\n", b"ERR 3\n"),
            (b"C 8 90\n", b"ERR 3\n"),
            (b"P\r\n", b"PONG\n"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                sim.receive(raw)
                self.assertEqual(drain(sim), [expected])

    def test_blank_lines_get_no_reply(self):
        sim, _ = booted_sim()
        sim.receive(b"\n   \r\n")
        self.assertEqual(drain(sim), [])

    def test_out_of_range_values_are_clamped_with_ok_c(self):
        sim, _ = booted_sim(**SPREAD_LIMITS)
        sim.receive(b"C 6 200\n")
        self.assertEqual(drain(sim), [b"OK C\n"])
        self.assertEqual(sim.target_deg[6], 120)
        sim.receive(b"C 6 100\n")
        self.assertEqual(drain(sim), [b"OK\n"])
        sim.receive(b"S 0 0 0 0 0 0 0 0\n")
        self.assertEqual(drain(sim), [b"OK C\n"])
        self.assertEqual(sim.target_deg[6], 60)
        sim.receive(b"V 99\n")
        self.assertEqual(drain(sim), [b"OK C\n"])
        self.assertEqual(sim.slew_deg, 30)

    def test_servos_slew_at_most_the_limit_per_tick(self):
        sim, clock = booted_sim()
        sim.receive(b"S 0 90 90 90 90 90 90 180\n")
        clock.advance(0.020)
        self.assertEqual(sim.angles_deg, (84, 90, 90, 90, 90, 90, 90, 96))
        clock.advance(0.020 * 14)
        self.assertEqual(sim.angles_deg, (0, 90, 90, 90, 90, 90, 90, 180))

    def test_watchdog_is_disarmed_until_the_first_command(self):
        sim, clock = booted_sim()
        clock.advance(10)
        self.assertEqual(drain(sim), [])

    def test_watchdog_goes_to_rest_then_relaxes_once(self):
        sim, clock = booted_sim()
        sim.receive(b"S 0 0 0 0 0 0 0 0\n")
        drain(sim)
        clock.advance(2.9)
        self.assertEqual(drain(sim), [])
        clock.advance(0.2)
        self.assertEqual(drain(sim), [b"WDT\n"])
        self.assertEqual(sim.target_deg, REST)
        self.assertTrue(sim.pwm_on)             # driving back to REST...
        clock.advance(1.0)
        self.assertFalse(sim.pwm_on)            # ...then limp
        self.assertEqual(sim.angles_deg, REST)
        clock.advance(10)
        self.assertEqual(drain(sim), [])        # fires once until the next command
        sim.receive(b"C 0 45\n")                # motion re-energizes, no re-arm needed
        self.assertEqual(drain(sim), [b"OK\n"])
        self.assertTrue(sim.pwm_on)

    def test_wdt_is_printed_before_the_reply_that_arrives_late(self):
        sim, clock = booted_sim()
        sim.receive(b"P\n")
        drain(sim)
        clock.advance(3.5)
        sim.receive(b"P\n")
        self.assertEqual(drain(sim), [b"WDT\n", b"PONG\n"])

    def test_lines_that_do_not_parse_do_not_feed_the_watchdog(self):
        sim, clock = booted_sim()
        sim.receive(b"P\n")
        drain(sim)
        seen = []
        for _ in range(4):
            clock.advance(1.0)
            sim.receive(b"Z\n")
            seen += drain(sim)
        self.assertIn(b"WDT\n", seen)

    def test_estop_latches_until_home(self):
        sim, _ = booted_sim()
        sim.receive(b"S 0 0 0 0 0 0 0 0\n")
        drain(sim)
        sim.press_estop()
        self.assertFalse(sim.pwm_on)
        for raw, expected in [(b"S 9 9 9 9 9 9 9 9\n", b"ERR 4\n"), (b"C 0 9\n", b"ERR 4\n"),
                              (b"P\n", b"PONG\n"), (b"V 5\n", b"OK\n"), (b"X\n", b"OK\n")]:
            with self.subTest(raw=raw):
                sim.receive(raw)
                self.assertEqual(drain(sim), [expected])
        sim.receive(b"Q\n")
        self.assertTrue(drain(sim)[0].startswith(b"A "))
        self.assertFalse(sim.pwm_on)
        sim.receive(b"H\n")
        self.assertEqual(drain(sim), [b"OK\n"])
        self.assertTrue(sim.pwm_on)
        self.assertEqual(sim.target_deg, REST)
        sim.receive(b"S 9 9 9 9 9 9 9 9\n")
        self.assertEqual(drain(sim), [b"OK\n"])

    def test_relax_then_motion_reenergizes(self):
        sim, _ = booted_sim()
        sim.receive(b"X\n")
        self.assertEqual(drain(sim), [b"OK\n"])
        self.assertFalse(sim.pwm_on)
        sim.receive(b"C 0 45\n")
        self.assertEqual(drain(sim), [b"OK\n"])
        self.assertTrue(sim.pwm_on)

    def test_standalone_demo_build_shows_letters(self):
        sim, _ = booted_sim(standalone_letters={"A": LETTER_A})
        sim.receive(b"L A\n")
        self.assertEqual(drain(sim), [b"OK\n"])
        self.assertEqual(sim.target_deg, LETTER_A)
        sim.receive(b"L B\n")                  # not in the table
        self.assertEqual(drain(sim), [b"ERR 3\n"])

    def test_rejects_wrong_channel_counts(self):
        with self.assertRaises(ValueError):
            FirmwareSim(clock=FakeClock(), rest_deg=(90,) * (N_CH - 1))


def open_mock(**sim_kwargs):
    clock = FakeClock()
    sim = FirmwareSim(clock=clock, **sim_kwargs)
    link = MockLink(sim, clock=clock)
    link.open()
    return link, sim, clock


class TestMockLink(unittest.TestCase):

    def test_open_handshakes_on_ready(self):
        link, _, _ = open_mock()
        self.assertTrue(link.is_open)
        self.assertEqual(link.ready, Ready(version=1, channels=N_CH))

    def test_every_protocol_command(self):
        link, sim, clock = open_mock(standalone_letters={"A": LETTER_A})
        frame = ServoFrame((10, 20, 30, 40, 50, 60, 70, 80))
        self.assertEqual(link.send_frame(frame), Ok())                     # S
        self.assertEqual(link.set_channel(3, 145), Ok())                   # C
        self.assertEqual(link.set_slew(30), Ok())                          # V
        self.assertTrue(link.ping())                                       # P
        clock.advance(1.0)
        self.assertEqual(link.query(), (10, 20, 30, 145, 50, 60, 70, 80))  # Q
        self.assertEqual(link.request(ShowLetter("A")), Ok())              # L
        self.assertEqual(link.relax(), Ok())                               # X
        self.assertFalse(sim.pwm_on)
        self.assertEqual(link.home(), Ok())                                # H
        self.assertTrue(sim.pwm_on)
        self.assertEqual([type(c) for c in link.sent],
                         [SetAll, SetChannel, SetSlew, Ping, Query, ShowLetter, Relax, Home])
        self.assertEqual(link.frames, [frame.angles_deg])

    def test_clamped_value_is_reported(self):
        link, _, _ = open_mock(**SPREAD_LIMITS)
        self.assertEqual(link.set_channel(6, 180), Ok(clamped=True))

    def test_normal_build_rejects_letters(self):
        link, _, _ = open_mock()
        self.assertEqual(link.request(ShowLetter("A")), Err(ErrorCode.UNKNOWN_COMMAND))

    def test_estop_refuses_motion_until_home(self):
        link, sim, _ = open_mock()
        frame = ServoFrame(REST)
        sim.press_estop()
        self.assertEqual(link.send_frame(frame), Err(ErrorCode.ESTOP))
        self.assertTrue(link.ping())
        self.assertEqual(link.home(), Ok())
        self.assertEqual(link.send_frame(frame), Ok())

    def test_watchdog_event_after_silence(self):
        link, _, clock = open_mock()
        link.ping()
        clock.advance(3.1)
        self.assertEqual(link.poll_events(), [Watchdog()])
        self.assertEqual(link.poll_events(), [])

    def test_keepalive_pings_prevent_the_watchdog(self):
        link, sim, clock = open_mock()
        link.send_frame(ServoFrame(REST))
        for _ in range(10):
            clock.advance(KEEPALIVE_S)
            self.assertTrue(link.ping())
        self.assertEqual(link.poll_events(), [])
        self.assertTrue(sim.pwm_on)

    def test_waiting_unsolicited_line_is_an_event_not_a_reply(self):
        link, _, clock = open_mock()
        link.ping()
        clock.advance(3.5)          # WDT is printed before the next command arrives
        self.assertTrue(link.ping())
        self.assertEqual(link.poll_events(), [Watchdog()])

    def test_late_reply_is_dropped_not_taken_as_the_next_answer(self):
        link, sim, _ = open_mock()
        sim.inject(b"OK\n")         # e.g. an answer that missed its ack timeout
        self.assertTrue(link.ping())
        self.assertEqual(link.stale_replies, 1)

    def test_silent_device_times_out_and_recovers(self):
        link, _, _ = open_mock()
        link.silent = True
        with self.assertRaises(LinkTimeout):
            link.ping()
        self.assertTrue(link.is_open)  # a timeout is not a disconnect
        link.silent = False
        self.assertTrue(link.ping())

    def test_unplug_and_reconnect(self):
        link, _, _ = open_mock()
        link.plugged = False
        with self.assertRaises(LinkDisconnected):
            link.send_frame(ServoFrame(REST))
        self.assertFalse(link.is_open)
        with self.assertRaises(LinkDisconnected):
            link.reconnect()
        link.plugged = True
        link.reconnect()
        self.assertEqual(link.ready, Ready(version=1, channels=N_CH))
        self.assertTrue(link.ping())

    def test_closed_link_refuses_requests(self):
        link, _, _ = open_mock()
        link.close()
        with self.assertRaises(LinkDisconnected):
            link.ping()

    def test_bad_values_never_reach_the_wire(self):
        link, _, _ = open_mock()
        with self.assertRaises(ValueError):
            link.set_channel(0, 181)
        self.assertEqual(link.sent, [])

    def test_dead_device_fails_the_handshake(self):
        clock = FakeClock()
        link = MockLink(FirmwareSim(clock=clock), clock=clock)
        link.silent = True
        with self.assertRaises(HandshakeError):
            link.open()
        self.assertFalse(link.is_open)

    def test_context_manager_closes(self):
        with MockLink(clock=FakeClock()) as link:
            self.assertTrue(link.ping())
        self.assertFalse(link.is_open)


def serial_link(sim_kwargs=None, setup=None, **port_kwargs):
    """A SerialLink whose "pyserial" is a SimSerialPort around one FirmwareSim."""
    clock = FakeClock()
    sim = FirmwareSim(clock=clock, **(sim_kwargs or {}))
    ports = []

    def factory(device, baud, timeout_s):
        port = SimSerialPort(sim, clock, **port_kwargs)
        port.timeout = timeout_s
        if setup:
            setup(port)
        ports.append(port)
        return port

    return SerialLink("/dev/fake", serial_factory=factory, clock=clock), sim, clock, ports


class TestSerialLink(unittest.TestCase):

    def test_open_waits_for_ready(self):
        link, _, _, ports = serial_link()
        link.open()
        self.assertEqual(link.ready, Ready(version=1, channels=N_CH))
        self.assertEqual(link.device, "/dev/fake")
        self.assertEqual(bytes(ports[0].written), b"")  # READY was enough; no ping

    def test_open_pings_a_board_that_did_not_reset(self):
        link, sim, clock, ports = serial_link(resets_on_open=False)
        sim.read(1024)  # its READY was printed long before we opened the port
        started = clock()
        link.open()
        self.assertTrue(link.is_open)
        self.assertIsNone(link.ready)
        self.assertEqual(bytes(ports[0].written), b"P\n")
        self.assertGreaterEqual(clock() - started + EPS, READY_TIMEOUT_S)

    def test_open_refuses_other_firmware(self):
        link, _, _, _ = serial_link(sim_kwargs={"version": 2})
        with self.assertRaises(HandshakeError):
            link.open()
        self.assertFalse(link.is_open)

    def test_open_gives_up_on_a_dead_port(self):
        link, sim, _, _ = serial_link(resets_on_open=False, setup=lambda p: setattr(p, "muted", True))
        sim.read(1024)
        with self.assertRaises(HandshakeError):
            link.open()
        self.assertFalse(link.is_open)

    def test_lines_split_across_reads(self):
        link, _, clock, _ = serial_link(chunk=1)
        link.open()
        self.assertEqual(link.ready, Ready(version=1, channels=N_CH))
        self.assertEqual(link.send_frame(ServoFrame((0,) * N_CH)), Ok())
        clock.advance(1.0)
        self.assertEqual(link.query(), (0,) * N_CH)

    def test_ack_timeout_is_about_50_ms(self):
        link, _, clock, ports = serial_link()
        link.open()
        ports[0].muted = True
        started = clock()
        with self.assertRaises(LinkTimeout):
            link.ping()
        waited = clock() - started
        self.assertGreaterEqual(waited + EPS, ACK_TIMEOUT_S)
        self.assertLess(waited, ACK_TIMEOUT_S + 2 * POLL_S)
        self.assertTrue(link.is_open)

    def test_reply_arriving_exactly_at_the_deadline_still_counts(self):
        link, _, _, ports = serial_link()
        link.open()
        ports[0].reply_delay_s = ACK_TIMEOUT_S
        self.assertTrue(link.ping())

    def test_reply_after_the_deadline_times_out_and_is_dropped_later(self):
        link, _, clock, ports = serial_link()
        link.open()
        ports[0].reply_delay_s = ACK_TIMEOUT_S + 0.02
        with self.assertRaises(LinkTimeout):
            link.ping()
        ports[0].reply_delay_s = 0.0
        clock.advance(0.1)             # the late PONG lands while nobody is waiting
        self.assertEqual(link.query(), REST)
        self.assertEqual(link.stale_replies, 1)

    def test_endless_line_noise_cannot_stall_a_request(self):
        class NoisyPort(SimSerialPort):
            @property
            def in_waiting(self):
                return 16

            def read(self, size=1):
                self.clock.advance(0.001)  # bytes take time on the wire
                return b"x" * size

        clock = FakeClock()
        sim = FirmwareSim(clock=clock)
        noisy = NoisyPort(sim, clock)

        def factory(device, baud, timeout_s):
            noisy.timeout = timeout_s
            return noisy

        link = SerialLink("/dev/fake", serial_factory=factory, clock=clock, ready_timeout_s=0.02)
        with self.assertLogs("hand.link", level="WARNING"), self.assertRaises(HandshakeError):
            link.open()  # no READY and no PONG through the noise: gives up in bounded time

    def test_unsolicited_and_garbled_lines_before_the_reply(self):
        link, _, _, ports = serial_link()
        link.open()
        ports[0].before_reply = b"WDT\n#?! noise\nREADY v1 ch=8\n"
        with self.assertLogs("hand.link", level="WARNING"):
            self.assertTrue(link.ping())
        self.assertEqual(link.poll_events(), [Watchdog(), Ready(version=1, channels=N_CH)])

    def test_late_reply_is_dropped(self):
        link, sim, _, _ = serial_link()
        link.open()
        sim.inject(b"OK\n")
        self.assertTrue(link.ping())
        self.assertEqual(link.stale_replies, 1)

    def test_line_noise_without_newline_is_discarded(self):
        link, sim, _, _ = serial_link()
        link.open()
        sim.inject(b"x" * 300)
        with self.assertLogs("hand.link", level="WARNING"):
            self.assertTrue(link.ping())

    def test_disconnect_then_reconnect(self):
        link, _, _, ports = serial_link()
        link.open()
        ports[0].broken = True
        with self.assertRaises(LinkDisconnected):
            link.send_frame(ServoFrame(REST))
        self.assertFalse(link.is_open)
        link.reconnect()               # a fresh port object, as after re-plugging
        self.assertEqual(len(ports), 2)
        self.assertTrue(link.ping())

    def test_open_failure_is_a_disconnect(self):
        def factory(device, baud, timeout_s):
            raise OSError(2, "No such file or directory")
        link = SerialLink("/dev/fake", serial_factory=factory, clock=FakeClock())
        with self.assertRaises(LinkDisconnected):
            link.open()

    def test_real_pyserial_errors_map_to_disconnect(self):
        """pyserial's SerialException is an OSError; the default factory relies on it."""
        link = SerialLink("/dev/collector-hand-does-not-exist")
        with self.assertRaises(LinkDisconnected):
            link.open()


class _Port:
    """The two ListPortInfo fields find_port reads."""

    def __init__(self, device, vid=None):
        self.device = device
        self.vid = vid


class TestFindPort(unittest.TestCase):

    def test_known_vendor_id_wins(self):
        ports = [_Port("/dev/cu.usbserial-X"), _Port("/dev/cu.debug", vid=0x2341)]
        self.assertEqual(find_port(ports), "/dev/cu.debug")

    def test_falls_back_to_port_name(self):
        ports = [_Port("/dev/cu.Bluetooth-Incoming-Port"), _Port("/dev/cu.usbmodem1101")]
        self.assertEqual(find_port(ports), "/dev/cu.usbmodem1101")

    def test_no_board_found(self):
        with self.assertRaises(LinkDisconnected):
            find_port([_Port("/dev/cu.Bluetooth-Incoming-Port")])


class TestLazyImport(unittest.TestCase):

    def test_links_work_without_importing_pyserial(self):
        code = ("import sys\n"
                "from hand.link import MockLink, SerialLink\n"
                "SerialLink('/dev/null')\n"
                "with MockLink() as link: link.ping()\n"
                "print('serial' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
