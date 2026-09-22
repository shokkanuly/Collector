"""Transport to the hand: one `Link` interface, two implementations (ARCHITECTURE.md §4).

- `MockLink` talks to `FirmwareSim`, an in-process model of the Arduino sketch.
  It records every frame, answers exactly as PROTOCOL.md says, runs the
  watchdog and E-stop logic, and can be made silent or "unplugged", so every
  failure path is testable with no Arduino (CLAUDE.md rule 7).
- `SerialLink` drives the real board over USB. pyserial is imported lazily in
  open() (CLAUDE.md rule 1), and tests can hand it any object that behaves
  like `serial.Serial` instead.

A link is not thread-safe: one owner (the HandController's hand thread) uses it.
"""
from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .protocol import (
    ACK_TIMEOUT_S, BAUD, READY_TIMEOUT_S, SLEW_DEFAULT_DEG, SLEW_MAX_DEG,
    SLEW_MIN_DEG, TICK_S, VERSION, WATCHDOG_RELAX_S, WATCHDOG_S, Angles, Command,
    Err, ErrorCode, Home, LineAssembler, LineTooLong, Ok, Ping, Pong, ProtocolError,
    Query, Ready, Relax, Reply, SetAll, SetChannel, SetSlew, ShowLetter, Watchdog,
    decode_command, decode_reply, encode_command, encode_reply, is_blank,
    is_unsolicited,
)
from .types import N_CH, SERVO_MAX_DEG, SERVO_MIN_DEG, ServoFrame

log = logging.getLogger(__name__)

Clock = Callable[[], float]


class LinkError(Exception):
    """Base class for transport failures."""


class LinkTimeout(LinkError):
    """No reply within the ack timeout."""


class LinkDisconnected(LinkError):
    """The port is closed, missing, or failed. reconnect() may bring it back."""


class HandshakeError(LinkError):
    """The device did not identify as protocol-v1 firmware with N_CH channels."""


class Link(ABC):
    """Send one command, receive its one reply (PROTOCOL.md §1).

    `READY` and `WDT` can arrive at any time and never answer a command:
    request() skips them, and poll_events() hands them to the caller. (After a
    READY the controller re-sends its frame; ARCHITECTURE.md §7.)
    """

    def __init__(self, *, ack_timeout_s: float, ready_timeout_s: float, clock: Clock):
        self.ack_timeout_s = ack_timeout_s
        self.ready_timeout_s = ready_timeout_s
        self._clock = clock
        self._events: Deque[Reply] = deque()
        self.ready: Optional[Ready] = None  # the handshake READY, if the board sent one
        self.stale_replies = 0              # late replies dropped before a new command

    # ---------------------------------------------------------------- transport hooks

    @property
    @abstractmethod
    def is_open(self) -> bool:
        """True while the transport is usable."""

    @abstractmethod
    def _connect(self) -> None:
        """Open the transport (the handshake happens afterwards, in open())."""

    @abstractmethod
    def _disconnect(self) -> None:
        """Close the transport. Must be safe to call when already closed."""

    @abstractmethod
    def _write(self, data: bytes) -> None:
        """Send bytes, raising LinkDisconnected if the transport failed."""

    @abstractmethod
    def _read_line(self, deadline: Optional[float]) -> Optional[bytes]:
        """The next complete line (newline included), or None.

        With a deadline (a clock() value), wait for a line until then. With
        None, return only a line that has already arrived and never block.
        """

    # ---------------------------------------------------------------- public API

    def open(self) -> None:
        """Connect and handshake. A no-op when already open."""
        if self.is_open:
            return
        self._connect()
        try:
            self._handshake()
        except BaseException:
            self._disconnect()
            raise

    def close(self) -> None:
        self._disconnect()

    def reconnect(self) -> None:
        self.close()
        self.open()

    def request(self, cmd: Command) -> Reply:
        """Send `cmd` and return its reply (OK, ERR, PONG, or A).

        Raises LinkTimeout when no reply arrives within the ack timeout and
        LinkDisconnected when the transport is gone. ERR replies are returned,
        not raised: E-stop (ERR 4) is a normal state for the caller to show.
        """
        data = encode_command(cmd)  # a bad value raises before anything is sent
        if not self.is_open:
            raise LinkDisconnected("link is not open")
        self._drain()
        self._write(data)
        deadline = self._clock() + self.ack_timeout_s
        while True:
            line = self._read_line(deadline)
            if line is None:
                raise LinkTimeout(f"no reply to {data!r} within {self.ack_timeout_s * 1000:.0f} ms")
            reply = self._decode(line)
            if reply is None:
                continue
            if is_unsolicited(reply):
                self._events.append(reply)
                continue
            return reply

    def poll_events(self) -> List[Reply]:
        """Unsolicited lines (READY, WDT) received so far, oldest first. Never blocks."""
        if self.is_open:
            self._drain()
        events = list(self._events)
        self._events.clear()
        return events

    def ping(self) -> bool:
        return isinstance(self.request(Ping()), Pong)

    def send_frame(self, frame: ServoFrame) -> Reply:
        return self.request(SetAll(frame.angles_deg))

    def set_channel(self, ch: int, angle_deg: int) -> Reply:
        return self.request(SetChannel(ch, angle_deg))

    def home(self) -> Reply:
        return self.request(Home())

    def relax(self) -> Reply:
        return self.request(Relax())

    def set_slew(self, deg_per_tick: int) -> Reply:
        return self.request(SetSlew(deg_per_tick))

    def query(self) -> Tuple[int, ...]:
        """Current (slewing) angles of all channels."""
        reply = self.request(Query())
        if not isinstance(reply, Angles):
            raise LinkError(f"Q was answered with {reply!r}")
        return reply.angles_deg

    def __enter__(self) -> "Link":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---------------------------------------------------------------- internals

    def _handshake(self) -> None:
        """Wait for READY; fall back to a ping for boards that did not reset."""
        self.ready = None
        deadline = self._clock() + self.ready_timeout_s
        while True:
            line = self._read_line(deadline)
            if line is None:
                break
            reply = self._decode(line)
            if isinstance(reply, Ready):
                if reply.version != VERSION or reply.channels != N_CH:
                    raise HandshakeError(f"device announced {reply}; this code speaks "
                                         f"protocol v{VERSION} with {N_CH} channels")
                self.ready = reply
                return
            # Anything before READY is boot noise: ignore it.
        try:
            alive = self.ping()
        except LinkTimeout:
            alive = False
        if not alive:
            raise HandshakeError(f"no READY within {self.ready_timeout_s} s and no PONG")

    def _drain(self) -> None:
        """Consume lines that arrived while nobody was waiting for them."""
        while True:
            line = self._read_line(None)
            if line is None:
                return
            reply = self._decode(line)
            if reply is None:
                continue
            if is_unsolicited(reply):
                self._events.append(reply)
            else:
                # A reply that missed its ack timeout. Dropping it here keeps it
                # from being taken as the answer to the next command.
                self.stale_replies += 1
                log.debug("dropped a late reply: %r", reply)

    @staticmethod
    def _decode(line: bytes) -> Optional[Reply]:
        try:
            return decode_reply(line)
        except ProtocolError as e:
            log.warning("ignoring a garbled line from the hand: %s", e)
            return None


# ---------------------------------------------------------------- firmware model

class FirmwareSim:
    """Behavioral model of firmware/collector_hand (PROTOCOL.md, ARCHITECTURE.md §6).

    It is the executable spec for the sketch: in the stage-3 bench test the real
    board should answer every line the way this model does. Time comes from
    `clock`, so tests can step it. Servo slew and the watchdog are evaluated
    lazily, in time order, whenever the model is touched.

    `standalone_letters` models a STANDALONE_DEMO build (letter -> 8 angles);
    None models a normal build, which answers `L` with ERR 1.
    """

    def __init__(self, clock: Clock = time.monotonic, *,
                 hard_min_deg: Sequence[int] = (SERVO_MIN_DEG,) * N_CH,
                 hard_max_deg: Sequence[int] = (SERVO_MAX_DEG,) * N_CH,
                 rest_deg: Sequence[int] = (90,) * N_CH,
                 standalone_letters: Optional[Mapping[str, Sequence[int]]] = None,
                 version: int = VERSION):
        for name, values in (("hard_min_deg", hard_min_deg), ("hard_max_deg", hard_max_deg),
                             ("rest_deg", rest_deg)):
            if len(values) != N_CH:
                raise ValueError(f"{name} needs {N_CH} values")
        self._clock = clock
        self.hard_min_deg = tuple(hard_min_deg)
        self.hard_max_deg = tuple(hard_max_deg)
        self.rest_deg = tuple(rest_deg)
        self.standalone_letters = (None if standalone_letters is None
                                   else {k: tuple(v) for k, v in standalone_letters.items()})
        self.version = version
        self.boot()

    def boot(self) -> None:
        """Power-on or reset: servos limp at REST, watchdog disarmed, then READY."""
        now = self._clock()
        self._asm = LineAssembler()
        self._out = bytearray()
        self._target: List[int] = list(self.rest_deg)
        self._current: List[int] = list(self.rest_deg)
        self._pwm_on = False
        self._estopped = False
        self._slew_deg = SLEW_DEFAULT_DEG
        self._boot_t = now
        self._ticks_done = 0
        self._last_cmd_t = now
        self._wdt_armed = False  # armed by the first valid command after boot
        self._relax_at: Optional[float] = None
        self._emit(Ready(self.version, N_CH))

    # -- the wire

    def receive(self, data: bytes) -> None:
        """Bytes from the host (Python -> Arduino)."""
        self._update()
        for item in self._asm.feed(data):
            if isinstance(item, LineTooLong):
                self._emit(Err(ErrorCode.LINE_TOO_LONG))
            elif not is_blank(item):
                self._emit(self._execute(item))

    @property
    def out_waiting(self) -> int:
        """Bytes waiting to be read by the host."""
        self._update()
        return len(self._out)

    def read(self, size: int) -> bytes:
        """Up to `size` bytes toward the host (Arduino -> Python)."""
        self._update()
        data = bytes(self._out[:size])
        del self._out[:size]
        return data

    def read_line(self) -> Optional[bytes]:
        """The next complete line toward the host, or None."""
        self._update()
        nl = self._out.find(b"\n")
        if nl < 0:
            return None
        line = bytes(self._out[:nl + 1])
        del self._out[:nl + 1]
        return line

    def inject(self, data: bytes) -> None:
        """Put raw bytes on the wire toward the host (tests: late or garbled lines)."""
        self._out += data

    # -- the hardware

    def press_estop(self) -> None:
        """The D2 button: PWM off at once, latched until `H`."""
        self._update()
        self._estopped = True
        self._pwm_on = False

    # Read-only state for tests. Each read replays pending time first, so what
    # it shows is the state at clock() now.

    @property
    def angles_deg(self) -> Tuple[int, ...]:
        """Current (slewing) servo angles."""
        self._update()
        return tuple(self._current)

    @property
    def target_deg(self) -> Tuple[int, ...]:
        self._update()
        return tuple(self._target)

    @property
    def pwm_on(self) -> bool:
        self._update()
        return self._pwm_on

    @property
    def estopped(self) -> bool:
        return self._estopped

    @property
    def slew_deg(self) -> int:
        return self._slew_deg

    # -- internals

    def _emit(self, reply: Reply) -> None:
        self._out += encode_reply(reply)

    def _execute(self, line: bytes) -> Reply:
        try:
            cmd = decode_command(line)
        except ProtocolError as e:
            return Err(e.code)
        if isinstance(cmd, ShowLetter) and self.standalone_letters is None:
            return Err(ErrorCode.UNKNOWN_COMMAND)  # a normal build has no L command

        # Every command that parses feeds the watchdog, even one the E-stop refuses.
        self._last_cmd_t = self._clock()
        self._wdt_armed = True
        self._relax_at = None

        if isinstance(cmd, Ping):
            return Pong()
        if isinstance(cmd, Query):
            return Angles(tuple(self._current))
        if isinstance(cmd, Relax):
            self._pwm_on = False
            return Ok()
        if isinstance(cmd, SetSlew):
            slew = min(max(cmd.deg_per_tick, SLEW_MIN_DEG), SLEW_MAX_DEG)
            self._slew_deg = slew
            return Ok(clamped=slew != cmd.deg_per_tick)
        if isinstance(cmd, Home):
            self._estopped = False
            self._target = list(self.rest_deg)
            self._pwm_on = True
            return Ok()
        if self._estopped:  # motion commands only
            return Err(ErrorCode.ESTOP)
        if isinstance(cmd, SetAll):
            return self._move(dict(enumerate(cmd.angles_deg)))
        if isinstance(cmd, SetChannel):
            return self._move({cmd.ch: cmd.angle_deg})
        # ShowLetter on a STANDALONE_DEMO build
        angles = self.standalone_letters.get(cmd.letter)
        if angles is None:
            return Err(ErrorCode.BAD_VALUE)
        return self._move(dict(enumerate(angles)))

    def _move(self, targets: Dict[int, int]) -> Ok:
        clamped = False
        for ch, angle in targets.items():
            limited = min(max(angle, self.hard_min_deg[ch]), self.hard_max_deg[ch])
            clamped = clamped or limited != angle
            self._target[ch] = limited
        self._pwm_on = True  # re-energizes after X or a watchdog relax
        return Ok(clamped=clamped)

    def _update(self) -> None:
        """Replay everything due up to now, in time order."""
        now = self._clock()
        if self._wdt_armed and now >= self._last_cmd_t + WATCHDOG_S:
            fired_t = self._last_cmd_t + WATCHDOG_S
            self._advance_servos(fired_t)
            self._wdt_armed = False
            self._target = list(self.rest_deg)
            self._relax_at = fired_t + WATCHDOG_RELAX_S
            self._emit(Watchdog())
        if self._relax_at is not None and now >= self._relax_at:
            self._advance_servos(self._relax_at)
            self._relax_at = None
            self._pwm_on = False
        self._advance_servos(now)

    def _advance_servos(self, t: float) -> None:
        # Count whole 20 ms ticks since boot; the epsilon absorbs float error
        # (0.06 / 0.02 must be 3 ticks, not 2.999...).
        tick = math.floor((t - self._boot_t) / TICK_S + 1e-9)
        ticks, self._ticks_done = tick - self._ticks_done, max(tick, self._ticks_done)
        if ticks <= 0 or not self._pwm_on:
            return
        step = self._slew_deg * ticks
        self._current = [c + max(-step, min(step, t_ - c)) for c, t_ in zip(self._current, self._target)]


class MockLink(Link):
    """A Link to FirmwareSim instead of a board. Records everything it sends.

    Fault injection: set `silent` and the device stops answering (requests time
    out), or set `plugged` to False to pull the USB cable (requests raise
    LinkDisconnected until reconnect() with the cable back in).
    """

    def __init__(self, sim: Optional[FirmwareSim] = None, *, clock: Clock = time.monotonic,
                 ack_timeout_s: float = ACK_TIMEOUT_S):
        # The model answers synchronously, so there is never anything to wait for.
        super().__init__(ack_timeout_s=ack_timeout_s, ready_timeout_s=0.0, clock=clock)
        self.sim = sim if sim is not None else FirmwareSim(clock=clock)
        self.sent: List[Command] = []
        self.frames: List[Tuple[int, ...]] = []  # the angles of every S sent
        self.silent = False
        self.plugged = True
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def _connect(self) -> None:
        self._check_plugged()
        self._open = True
        self.sim.boot()  # opening the port resets an Uno

    def _disconnect(self) -> None:
        self._open = False

    def _write(self, data: bytes) -> None:
        self._check_plugged()
        cmd = decode_command(data)  # our own encoder made it, so it always parses
        self.sent.append(cmd)
        if isinstance(cmd, SetAll):
            self.frames.append(cmd.angles_deg)
        if not self.silent:
            self.sim.receive(data)

    def _read_line(self, deadline: Optional[float]) -> Optional[bytes]:
        self._check_plugged()
        return None if self.silent else self.sim.read_line()

    def _check_plugged(self) -> None:
        if not self.plugged:
            self._open = False
            raise LinkDisconnected("mock hand is unplugged")


# ---------------------------------------------------------------- the real board

POLL_S = 0.005          # longest single blocking read; bounds how late a deadline is noticed
WRITE_TIMEOUT_S = 0.1
_MAX_PENDING = 256      # bytes without a newline before the RX buffer is dropped as noise

# USB vendor IDs of boards the firmware runs on: Arduino (2), CH340, FTDI, CP210x.
_BOARD_VIDS = {0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4}
_PORT_HINTS = ("usbmodem", "usbserial", "wchusbserial", "ttyACM", "ttyUSB")


def find_port(ports: Optional[Iterable[object]] = None) -> str:
    """Pick the Arduino's serial port for `port: auto` (config/hand.yaml).

    A known USB vendor ID wins over a name that merely looks like a board.
    """
    if ports is None:
        from serial.tools import list_ports  # lazy: CLAUDE.md rule 1
        ports = list_ports.comports()
    ports = list(ports)
    by_vid = [p for p in ports if getattr(p, "vid", None) in _BOARD_VIDS]
    by_name = [p for p in ports if any(h in p.device for h in _PORT_HINTS)]
    for port in by_vid + by_name:
        return port.device
    raise LinkDisconnected("no Arduino-like serial port found; "
                           "pass one explicitly, e.g. --hand /dev/tty.usbmodem1101")


def _open_pyserial(device: str, baud: int, timeout_s: float) -> object:
    import serial  # lazy: CLAUDE.md rule 1
    # pyserial defaults to 8N1 with no flow control, as PROTOCOL.md §1 requires.
    return serial.Serial(port=device, baudrate=baud, timeout=timeout_s, write_timeout=WRITE_TIMEOUT_S)


class SerialLink(Link):
    """The real board over USB serial.

    `serial_factory(device, baud, timeout_s)` opens the port; it defaults to
    pyserial and is imported only when open() runs. Every pyserial error is an
    OSError, which this class turns into LinkDisconnected.
    """

    def __init__(self, port: str = "auto", baud: int = BAUD, *,
                 ack_timeout_s: float = ACK_TIMEOUT_S,
                 ready_timeout_s: float = READY_TIMEOUT_S,
                 serial_factory: Optional[Callable[[str, int, float], object]] = None,
                 clock: Clock = time.monotonic):
        super().__init__(ack_timeout_s=ack_timeout_s, ready_timeout_s=ready_timeout_s, clock=clock)
        self.port = port
        self.baud = baud
        self.device: Optional[str] = None  # the resolved port once opened
        self._factory = serial_factory or _open_pyserial
        self._ser: Optional[object] = None
        self._rx = bytearray()

    @property
    def is_open(self) -> bool:
        return self._ser is not None and bool(getattr(self._ser, "is_open", True))

    def _connect(self) -> None:
        device = find_port() if self.port == "auto" else self.port
        try:
            self._ser = self._factory(device, self.baud, POLL_S)
        except OSError as e:
            raise LinkDisconnected(f"cannot open {device}: {e}") from e
        self.device = device
        self._rx.clear()

    def _disconnect(self) -> None:
        ser, self._ser = self._ser, None
        if ser is not None:
            try:
                ser.close()
            except OSError:
                pass

    def _write(self, data: bytes) -> None:
        try:
            self._ser.write(data)
            self._ser.flush()
        except OSError as e:
            raise self._lost(e) from e

    def _read_line(self, deadline: Optional[float]) -> Optional[bytes]:
        while True:
            nl = self._rx.find(b"\n")
            if nl >= 0:
                line = bytes(self._rx[:nl + 1])
                del self._rx[:nl + 1]
                return line
            try:
                waiting = self._ser.in_waiting
                if deadline is None:
                    if not waiting:
                        return None
                elif self._clock() >= deadline:
                    return None
                chunk = self._ser.read(max(1, waiting))  # blocks at most POLL_S
            except OSError as e:
                raise self._lost(e) from e
            self._rx += chunk
            if len(self._rx) > _MAX_PENDING:
                log.warning("dropping %d bytes of line noise from %s", len(self._rx), self.device)
                self._rx.clear()

    def _lost(self, error: OSError) -> LinkDisconnected:
        self._disconnect()
        return LinkDisconnected(f"serial port {self.device} failed: {error}")
