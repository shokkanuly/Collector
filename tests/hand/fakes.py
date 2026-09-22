"""Test doubles shared by the hand tests: a steppable clock and a fake serial port."""
from typing import Optional

from hand.link import FirmwareSim


class FakeClock:
    """A monotonic clock that only moves when told to."""

    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class SimSerialPort:
    """Stands in for serial.Serial, wired to a FirmwareSim.

    resets_on_open  an Uno auto-resets (and prints READY) when the port opens
    chunk           max bytes per read(), to exercise lines split across reads
    muted           the device stops answering: written bytes vanish
    broken          the cable is gone: every call raises OSError, like pyserial
    before_reply    bytes the device emits right before answering the next command
    """

    def __init__(self, sim: FirmwareSim, clock: FakeClock, *, resets_on_open: bool = True,
                 chunk: Optional[int] = None):
        self.sim = sim
        self.clock = clock
        self.chunk = chunk
        self.timeout = 0.005
        self.is_open = True
        self.muted = False
        self.broken = False
        self.before_reply = b""
        self.written = bytearray()
        if resets_on_open:
            sim.boot()

    @property
    def in_waiting(self) -> int:
        self._check()
        return self.sim.out_waiting

    def write(self, data: bytes) -> int:
        self._check()
        self.written += data
        if not self.muted:
            self.sim.inject(self.before_reply)
            self.before_reply = b""
            self.sim.receive(bytes(data))
        return len(data)

    def flush(self) -> None:
        self._check()

    def read(self, size: int = 1) -> bytes:
        self._check()
        data = self.sim.read(min(size, self.chunk or size))
        if not data:
            self.clock.advance(self.timeout)  # a real port blocks for its timeout
        return data

    def close(self) -> None:
        self.is_open = False

    def _check(self) -> None:
        if self.broken:
            raise OSError("device reports readiness to read but returned no data")
        if not self.is_open:
            raise OSError("port is closed")
