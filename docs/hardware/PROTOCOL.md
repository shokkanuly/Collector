# Collector Hand: Serial Protocol v1

The contract between `hand/protocol.py` and `firmware/collector_hand`. Any change must update both sides and their tests in the same commit (CLAUDE.md rule 6).

## 1. Transport

- USB CDC serial, **115200 baud, 8N1**, no flow control.
- ASCII, one command per line, terminated by `\n` (a trailing `\r` is ignored).
- Fields are separated by one or more spaces; leading and trailing spaces are ignored. Command letters are upper case. Blank lines are ignored and get no reply.
- An integer is an optional `-` followed by 1–4 ASCII digits (`7`, `007`, `-5`). Anything else (`+5`, `5.0`, `0x10`, `12345`) is `ERR 3`. The 4-digit cap keeps every value inside an AVR `int16_t`.
- Max line length is **48 bytes** including `\n` (a trailing `\r` counts). Longer lines are discarded up to their `\n` and answered once with `ERR 2`.
- Every command gets exactly one reply line. Python waits ≤ 50 ms for it (the ack timeout).
- After reset the Arduino prints `READY v1 ch=8`. The Python link waits for it (≤ 2.5 s, because the Uno auto-resets when the port opens).

## 2. Commands (Python → Arduino)

| Command | Example | Meaning | Reply |
|---|---|---|---|
| `S a0 a1 … a7` | `S 90 120 0 0 0 0 90 90` | Set the target angle (integer degrees 0–180) for all 8 channels. The firmware slews toward it. | `OK` |
| `C ch a` | `C 3 145` | Set one channel only (calibration). | `OK` |
| `H` | `H` | Home: go to REST pose (open hand, thumb out, wrist 90) and re-enable PWM after an E-stop. | `OK` |
| `X` | `X` | Relax: PWM off on all channels (servos go limp). | `OK` |
| `P` | `P` | Ping. | `PONG` |
| `Q` | `Q` | Query current angles. | `A 90 118 3 0 …` (8 values) |
| `V s` | `V 6` | Set slew limit in degrees per 20 ms tick (1–30). | `OK` |

Angles outside `[HARD_MIN[ch], HARD_MAX[ch]]` are **clamped, not rejected**. The reply is then `OK C` so Python can log that its config is wider than the firmware allows. A slew value outside 1–30 is clamped the same way (`OK C`).

A channel number outside 0–7 is **rejected** with `ERR 3`, because clamping it would move the wrong servo.

Python never sends an angle outside 0–180: `hand/protocol.py` refuses to encode one, so clamping only ever triggers on the firmware's narrower `HARD_MIN`/`HARD_MAX` or on lines typed by hand.

## 3. Replies and errors (Arduino → Python)

| Reply | Meaning |
|---|---|
| `OK` / `OK C` | accepted / accepted with clamping |
| `ERR 1` | unknown command or wrong number of fields |
| `ERR 2` | line too long |
| `ERR 3` | value not an integer, or channel out of range |
| `ERR 4` | E-stop active: send `H` to re-arm |
| `READY v1 ch=8` | unsolicited, after boot/reset |
| `WDT` | unsolicited: watchdog fired (no command for 3 s), hand relaxed |

When several errors apply, the first failing check wins: line length (`ERR 2`), then command letter and field count (`ERR 1`), then values (`ERR 3`), then E-stop (`ERR 4`).

`READY` and `WDT` can arrive at any time, including just before a reply. They never count as the reply to a command.

## 4. Timing

- Python sends at most **30 frames/s** (`S` lines are ≤ 34 bytes, about 1 KB/s, far below the line capacity).
- Firmware servo tick: 20 ms (50 Hz, matching servo PWM).
- Watchdog: 3 s without a valid command → REST, then PWM off 1 s later. In letter mode, Python sends `P` every 1 s while idle to keep the hand alive.

## 5. Standalone demo (optional)

`scripts/build_poses.py --emit-firmware-table` writes `firmware/collector_hand/poses_table.h` with the 26 letters as a `PROGMEM` array. With `#define STANDALONE_DEMO`, the sketch also accepts `L <letter>` → `OK`, for demos without a laptop. The table is generated from `hand/poses.json` and never edited by hand. A demo build answers a field that is not a single letter A–Z with `ERR 3`; a normal build answers any `L` line with `ERR 1`.

## 6. Example session

The angles are illustrative. Real values depend on the calibration and `invert` flags in `config/hand.yaml`.

```
← READY v1 ch=8
→ P                      ← PONG
→ H                      ← OK
→ S 25 140 150 170 170 150 90 90     (A)   ← OK
→ S 140 60 0 0 0 0 90 90             (B)   ← OK
→ C 6 200                ← OK C       (clamped to HARD_MAX[6])
(3 s silence)            ← WDT
```
