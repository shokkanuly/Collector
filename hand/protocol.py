"""Serial protocol v1 codec (docs/hardware/PROTOCOL.md). Pure functions, no I/O.

Both directions exist for both kinds of line. The Python link encodes commands
and decodes replies; MockLink's firmware model decodes commands and encodes
replies. Keeping all four here means the mock speaks exactly the bytes the real
firmware must, and the round-trip tests pin the format.

Any change must land in the same commit as PROTOCOL.md, the firmware parser,
and their tests (CLAUDE.md rule 6).
"""
from __future__ import annotations

import operator
import re
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple, Union

from .types import N_CH, SERVO_MAX_DEG, SERVO_MIN_DEG

# §1 Transport
VERSION = 1
BAUD = 115200
MAX_LINE_BYTES = 48          # including the trailing "\n"
ACK_TIMEOUT_S = 0.050        # how long Python waits for each reply
READY_TIMEOUT_S = 2.5        # the Uno auto-resets when the port opens

# §4 Timing
MAX_RATE_HZ = 30
TICK_S = 0.020               # firmware servo tick (50 Hz)
WATCHDOG_S = 3.0             # no valid command for this long -> REST
WATCHDOG_RELAX_S = 1.0       # ...then PWM off this much later
KEEPALIVE_S = 1.0            # letter mode pings this often while idle

# §2 `V s`: slew limit in degrees per tick
SLEW_MIN_DEG = 1
SLEW_MAX_DEG = 30
SLEW_DEFAULT_DEG = 6

# ASCII digits only: str patterns with \d would also accept other Unicode digits.
_INT = re.compile(r"-?[0-9]{1,4}")
_READY = re.compile(r"READY v([0-9]+) ch=([0-9]+)")


class ErrorCode(IntEnum):
    """The number in an `ERR n` reply (PROTOCOL.md §3)."""

    UNKNOWN_COMMAND = 1   # unknown command or wrong number of fields
    LINE_TOO_LONG = 2
    BAD_VALUE = 3         # value not an integer, or channel out of range
    ESTOP = 4             # E-stop active: send H to re-arm


class ProtocolError(ValueError):
    """A line that does not parse.

    For command lines, `code` is the ERR the firmware answers with. Reply lines
    have no such code (Python never answers the firmware), so it is None there.
    """

    def __init__(self, message: str, code: Optional[ErrorCode] = None):
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------- commands (Python -> Arduino)
# Plain records: the encoder validates what Python may send, while the decoder
# accepts any integer, because the firmware clamps out-of-range values.

@dataclass(frozen=True)
class SetAll:
    """`S a0 .. a7`: target angle for every channel."""

    angles_deg: Tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "angles_deg", tuple(self.angles_deg))


@dataclass(frozen=True)
class SetChannel:
    """`C ch a`: target angle for one channel (calibration)."""

    ch: int
    angle_deg: int


@dataclass(frozen=True)
class Home:
    """`H`: go to REST and re-arm after an E-stop."""


@dataclass(frozen=True)
class Relax:
    """`X`: PWM off on all channels."""


@dataclass(frozen=True)
class Ping:
    """`P`: liveness check, answered with PONG."""


@dataclass(frozen=True)
class Query:
    """`Q`: current angles, answered with `A a0 .. a7`."""


@dataclass(frozen=True)
class SetSlew:
    """`V s`: slew limit in degrees per 20 ms tick."""

    deg_per_tick: int


@dataclass(frozen=True)
class ShowLetter:
    """`L <letter>`: only on firmware built with STANDALONE_DEMO (PROTOCOL.md §5)."""

    letter: str


Command = Union[SetAll, SetChannel, Home, Relax, Ping, Query, SetSlew, ShowLetter]

# Command letter -> number of fields after it.
_ARITY = {"S": N_CH, "C": 2, "H": 0, "X": 0, "P": 0, "Q": 0, "V": 1, "L": 1}
_NO_ARGS = {"H": Home, "X": Relax, "P": Ping, "Q": Query}


# ---------------------------------------------------------------- replies (Arduino -> Python)

@dataclass(frozen=True)
class Ok:
    """`OK`, or `OK C` when the firmware clamped a value."""

    clamped: bool = False


@dataclass(frozen=True)
class Err:
    """`ERR n`."""

    code: ErrorCode


@dataclass(frozen=True)
class Pong:
    """`PONG`, the answer to `P`."""


@dataclass(frozen=True)
class Angles:
    """`A a0 .. a7`: current (slewing) angles, the answer to `Q`."""

    angles_deg: Tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "angles_deg", tuple(self.angles_deg))


@dataclass(frozen=True)
class Ready:
    """`READY v1 ch=8`: unsolicited, printed after boot/reset."""

    version: int
    channels: int


@dataclass(frozen=True)
class Watchdog:
    """`WDT`: unsolicited, the watchdog fired and the hand relaxed."""


Reply = Union[Ok, Err, Pong, Angles, Ready, Watchdog]


def is_unsolicited(reply: Reply) -> bool:
    """True for lines the firmware sends on its own; they never answer a command."""
    return isinstance(reply, (Ready, Watchdog))


# ---------------------------------------------------------------- encoding

def _int_in(name: str, value: object, lo: int, hi: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {value!r}")
    value = operator.index(value)  # accepts numpy ints, rejects floats
    if not lo <= value <= hi:
        raise ValueError(f"{name}={value} is outside [{lo}, {hi}]")
    return value


def _angles(values: Tuple[int, ...]) -> List[int]:
    if len(values) != N_CH:
        raise ValueError(f"need {N_CH} angles, got {len(values)}")
    return [_int_in(f"angle[{ch}]", a, SERVO_MIN_DEG, SERVO_MAX_DEG) for ch, a in enumerate(values)]


def _line(text: str) -> bytes:
    line = (text + "\n").encode("ascii")
    if len(line) > MAX_LINE_BYTES:
        raise ValueError(f"encoded line is {len(line)} bytes; the limit is {MAX_LINE_BYTES}")
    return line


def encode_command(cmd: Command) -> bytes:
    """One command line, including the trailing newline.

    Strict: Python never sends a value outside the wire range, so a bad value
    here is a bug upstream and raises instead of relying on firmware clamping.
    """
    if isinstance(cmd, SetAll):
        return _line("S " + " ".join(map(str, _angles(cmd.angles_deg))))
    if isinstance(cmd, SetChannel):
        ch = _int_in("ch", cmd.ch, 0, N_CH - 1)
        angle = _int_in("angle_deg", cmd.angle_deg, SERVO_MIN_DEG, SERVO_MAX_DEG)
        return _line(f"C {ch} {angle}")
    if isinstance(cmd, SetSlew):
        return _line(f"V {_int_in('deg_per_tick', cmd.deg_per_tick, SLEW_MIN_DEG, SLEW_MAX_DEG)}")
    if isinstance(cmd, ShowLetter):
        if not (isinstance(cmd.letter, str) and len(cmd.letter) == 1 and "A" <= cmd.letter <= "Z"):
            raise ValueError(f"letter must be one of A-Z, got {cmd.letter!r}")
        return _line(f"L {cmd.letter}")
    for name, cls in _NO_ARGS.items():
        if isinstance(cmd, cls):
            return _line(name)
    raise TypeError(f"not a protocol command: {cmd!r}")


def encode_reply(reply: Reply) -> bytes:
    """One reply line, as the firmware prints it."""
    if isinstance(reply, Ok):
        return _line("OK C" if reply.clamped else "OK")
    if isinstance(reply, Err):
        return _line(f"ERR {int(ErrorCode(reply.code))}")
    if isinstance(reply, Pong):
        return _line("PONG")
    if isinstance(reply, Angles):
        return _line("A " + " ".join(map(str, _angles(reply.angles_deg))))
    if isinstance(reply, Ready):
        return _line(f"READY v{int(reply.version)} ch={int(reply.channels)}")
    if isinstance(reply, Watchdog):
        return _line("WDT")
    raise TypeError(f"not a protocol reply: {reply!r}")


# ---------------------------------------------------------------- decoding

def _text(line: Union[bytes, str]) -> str:
    """Drop the line terminator: "\\n", optionally preceded by "\\r"."""
    if isinstance(line, (bytes, bytearray)):
        # latin-1 maps every byte to one char, so a stray non-ASCII byte fails
        # the token checks below (ERR 1 or 3) exactly as it would on the firmware.
        line = bytes(line).decode("latin-1")
    if line.endswith("\n"):
        line = line[:-1]
    if line.endswith("\r"):
        line = line[:-1]
    return line


def split_fields(line: Union[bytes, str]) -> List[str]:
    """Tokenize like the firmware: fields split on runs of spaces, edges trimmed."""
    return [f for f in _text(line).split(" ") if f]


def is_blank(line: Union[bytes, str]) -> bool:
    """Blank lines are ignored by the firmware and get no reply."""
    return not split_fields(line)


def _parse_int(token: str) -> int:
    if not _INT.fullmatch(token):
        raise ProtocolError(f"not an integer: {token!r}", ErrorCode.BAD_VALUE)
    return int(token)


def decode_command(line: Union[bytes, str]) -> Command:
    """Parse one command line exactly as the firmware does.

    Raises ProtocolError carrying the ERR code the firmware answers with. Values
    outside the servo range are returned as they are; clamping (and the `OK C`
    answer) is the receiver's job. Line length is checked by framing, not here
    (see LineAssembler).
    """
    fields = split_fields(line)
    if not fields:
        raise ProtocolError("blank line", ErrorCode.UNKNOWN_COMMAND)
    name, args = fields[0], fields[1:]
    if _ARITY.get(name) != len(args):
        raise ProtocolError(f"unknown command or wrong field count: {' '.join(fields)!r}",
                            ErrorCode.UNKNOWN_COMMAND)
    if name == "L":
        letter = args[0]
        if len(letter) != 1 or not "A" <= letter <= "Z":
            raise ProtocolError(f"not a letter: {letter!r}", ErrorCode.BAD_VALUE)
        return ShowLetter(letter)
    values = [_parse_int(a) for a in args]
    if name == "S":
        return SetAll(tuple(values))
    if name == "C":
        ch, angle = values
        if not 0 <= ch < N_CH:
            raise ProtocolError(f"channel {ch} is outside 0-{N_CH - 1}", ErrorCode.BAD_VALUE)
        return SetChannel(ch, angle)
    if name == "V":
        return SetSlew(values[0])
    return _NO_ARGS[name]()


def decode_reply(line: Union[bytes, str]) -> Reply:
    """Parse one line from the firmware. Raises ProtocolError on anything else."""
    text = _text(line)
    fixed = {"OK": Ok(), "OK C": Ok(clamped=True), "PONG": Pong(), "WDT": Watchdog()}
    if text in fixed:
        return fixed[text]
    fields = text.split(" ")
    if fields[0] == "ERR" and len(fields) == 2 and _INT.fullmatch(fields[1]):
        try:
            return Err(ErrorCode(int(fields[1])))
        except ValueError:
            raise ProtocolError(f"unknown error code in {text!r}") from None
    if fields[0] == "A" and len(fields) == N_CH + 1 and all(_INT.fullmatch(f) for f in fields[1:]):
        return Angles(tuple(int(f) for f in fields[1:]))
    match = _READY.fullmatch(text)
    if match:
        return Ready(version=int(match.group(1)), channels=int(match.group(2)))
    raise ProtocolError(f"unrecognized reply {text!r}")


# ---------------------------------------------------------------- framing

@dataclass(frozen=True)
class LineTooLong:
    """Framing event: a line over MAX_LINE_BYTES was discarded (answered with ERR 2)."""


class LineAssembler:
    """Split a byte stream into lines the way the firmware does.

    The firmware keeps a fixed buffer of MAX_LINE_BYTES - 1 content bytes (the
    newline is the last of the 48). A longer line is dropped up to its newline
    and reported once as LineTooLong. A trailing "\\r" counts toward the limit.
    """

    def __init__(self, max_line_bytes: int = MAX_LINE_BYTES):
        self.max_content = max_line_bytes - 1
        self._buf = bytearray()
        self._overflow = False

    def feed(self, data: bytes) -> List[Union[bytes, LineTooLong]]:
        """Consume bytes; return each completed line (without "\\n") or LineTooLong."""
        out: List[Union[bytes, LineTooLong]] = []
        for byte in data:
            if byte == 0x0A:  # "\n"
                out.append(LineTooLong() if self._overflow else bytes(self._buf))
                self._buf.clear()
                self._overflow = False
            elif self._overflow:
                continue
            elif len(self._buf) < self.max_content:
                self._buf.append(byte)
            else:
                self._overflow = True
                self._buf.clear()
        return out
