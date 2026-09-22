# Collector Hand: Architecture

Status: **stages 0–2 implemented** (Python side, no hardware needed; see `docs/hardware/ROADMAP.md`). Stage 3 onward waits for hardware. Owner: shokkanuly. Last updated 2026-09-22.

## 1. Goal

Add a physical, human-proportioned robotic hand to Collector that can:

1. **Letter mode:** sign the letter the recognizer just read (webcam → NN → letter → hand).
2. **Spell mode:** fingerspell typed text (`hand_cli.py spell HELLO`), with no camera.
3. **Mirror mode:** copy the user's live hand pose frame by frame (webcam → landmarks → hand). No classifier is used.

Non-goals for this milestone: J/Z motion (it waits for the Stage 5 sequence model, although the wrist can fake it later), two hands, force sensing, and wireless.

## 2. System overview

```
┌──────────────────────── Laptop (Python) ────────────────────────┐        ┌──────── Arduino Uno ────────┐
│                                                                  │        │                              │
│ Webcam → HandTracker (MediaPipe) → 63 floats ─┬─► FeedForwardNN  │        │  Serial parser (PROTOCOL.md) │
│                                               │      │ letter    │        │          │                   │
│                                  mirror mode  │      ▼           │        │  Safety: clamp, slew limit,  │
│                                               │  PoseLibrary     │        │  watchdog → relax            │
│                                               ▼      │ HandPose  │  USB   │          │ I²C               │
│                                     Kinematics ──────┤           │ serial │  PCA9685 16-ch PWM driver    │
│                                               HandPose           │───────►│          │ PWM ×8            │
│                                                      ▼           │115200  │  8 servos → tendons → fingers│
│                                     ServoMapper (config/hand.yaml)│  baud  │                              │
│                                                      ▼           │        └──────────────────────────────┘
│                                     HandController (thread, 30 Hz, latest-wins)                            │
│                                                      ▼                                                     │
│                                     SerialLink | MockLink                                                  │
└──────────────────────────────────────────────────────────────────┘
```

Division of work: **the laptop decides what pose to make, and the Arduino makes the pose safely.** All intelligence (recognition, pose table, kinematics, calibration math) stays in Python, where it is testable. The firmware is a small, dumb, safe servo server.

## 3. Core data types (`hand/types.py`)

```python
@dataclass(frozen=True)
class HandPose:              # hardware-independent, all values normalized
    curl: tuple[float, float, float, float]   # index, middle, ring, pinky; 0 = straight, 1 = full fist
    thumb_flex: float        # 0 = straight, 1 = fully bent
    thumb_opp: float         # 0 = out to the side (L, Y), 1 = across palm toward pinky (M, B); A sits between
    spread: float            # index–middle abduction; 0 = together (U), 1 = wide (V)
    wrist_roll: float        # -1 .. 1; 0 = palm facing viewer; ±1 = ±90°
    
@dataclass(frozen=True)
class ServoFrame:            # hardware-specific
    angles_deg: tuple[int, ...]   # length N_CH (8), already clamped and calibrated
```

`HandPose` is the interface between the software and hardware halves. Everything upstream (NN, dataset, mirror) produces a `HandPose`, and everything downstream (mapper, link, firmware) consumes one. This keeps the hand swappable. A different mechanical hand only needs a new `config/hand.yaml`.

## 4. Python modules (`hand/`)

| Module | Responsibility | Depends on |
|---|---|---|
| `types.py` | `HandPose`, `ServoFrame`, constants (landmark indices) | none |
| `kinematics.py` | `pose_from_landmarks(np.ndarray[63]) -> HandPose` | numpy |
| `pose_library.py` | load `poses.json`; `pose_for(letter) -> HandPose`; open/rest poses | types |
| `handshapes.py` | ASL handshape per letter (each finger extended/closed/partial, thumb and spread state); shapes dataset poses and checks naturalness | types |
| `mapping.py` | `ServoMapper(config).to_frame(HandPose) -> ServoFrame` (calibration, inversion, clamping) | yaml |
| `protocol.py` | encode and decode PROTOCOL.md lines; pure functions, no I/O | none |
| `link.py` | `SerialLink(port)` and `MockLink()` behind one `Link` interface; reconnect, ack timeout | pyserial (lazy) |
| `controller.py` | `HandController`: thread, latest-wins queue, 30 Hz cap, smoothing (EMA), hold-time for letters, modes | all above |
| `scripts/build_poses.py` | dataset → `hand/poses.json` + `reports/poses_report.md` | pandas |
| `scripts/hand_cli.py` | `pose A`, `spell HELLO`, `sweep 2`, `calibrate`, `raw 90 90 ...`, `--mock` | controller |

### 4.1 Kinematics: landmarks → HandPose

Input: the 63 normalized floats already produced by `HandTracker.extract_landmarks`. MediaPipe indices: wrist 0; thumb 1–4; index 5–8; middle 9–12; ring 13–16; pinky 17–20.

- **Finger curl**: the sum of the three bend angles at MCP, PIP, and DIP. The bend at joint B between A→B and B→C is `acos(dot(B−A, C−B))`. The MCP bend uses wrist→MCP as the reference segment. Normalize with dataset percentiles (p2 ≈ 10°, p98 ≈ 270°) → `curl_01`.
- **Thumb flex**: bend at landmark 2 plus bend at landmark 3. Normalize with p2 ≈ 15°, p98 ≈ 130°.
- **Thumb opposition**: distance from the thumb tip (4) to the pinky MCP (17), in the normalized scale. Dataset range is 0.2 (across the palm, M) to 1.8 (out, L). Map inversely.
- **Spread**: the angle between the index proximal phalanx (5→6) and the middle proximal phalanx (9→10). Using fingertip vectors instead gets confused by curl; see §9.
- **Wrist roll**: the angle of the wrist→middle-MCP vector (0→9) in the image plane. Only used in mirror mode. Letters G/H/P/Q take their orientation from overrides (§9).

All five functions are pure and vectorized (they accept `(N,63)`), so `build_poses.py` and the tests run them over the whole dataset.

**As built in stage 2** (`hand/kinematics.py`). Three changes from the sketch above, each forced by the dataset:

- **Curl uses one range for all four fingers, anchored on what the letters mean:** 17.4° (the median finger that ASL extends) → 0, and 240.9° (the median finger of the A and S fists) → 1. The same bend then reads the same on every finger, so partial shapes stay even: C is 0.60–0.66 on all four fingers. *Revised in the naturalness pass:* stage 2 first used per-finger p2/p98 ranges, which read the index about 30% more curled than the other fingers at the same angle and made C and O lopsided. The ranges' one advantage, letting the index close fully, is now handled by shaping (§4.2). The p2/p98 of all fingers pooled (7.7/303.9°) was never an option: it drops T's index to 0.59, below the gate.
- **Spread is normalized over frames where index and middle are both extended** (p2/p98 = 1.1/17.1°), and it fades to 0 as either finger curls (`curl_01` 0.25 → 0.45, inside the gap between letters with both fingers extended, median at most 0.20 for H, and letters with one curled, at least 0.59 for O). When one finger is curled, the angle between the proximal phalanges measures flexion, not abduction: G/P/Q/X read 70–120°. A range over all frames therefore squashed U and V to 0.04 apart, while the extended-only range gives U 0.10 and V 0.42. The U and V session medians never overlap (U ≤ 5.2°, V ≥ 6.1°), so the metric itself works; this signer's V is simply narrow.
- **Wrist roll comes from the palm normal**, `atan2(n_x, n_z)` with `n = (5−0) × (17−0)`, where 0 means the palm faces the camera. `HandPose.wrist_roll` is defined by where the palm faces. The in-plane angle of 0→9 measures tilt within the image and is blind to roll. In the data, H's palm faces sideways (median normal x 0.72), which the palm normal detects.

Also: angles use `atan2(|u×v|, u·v)`, not `acos`, because `acos` of a rounded cosine is inaccurate near 0°, exactly where straight fingers sit. The ranges are frozen in `kinematics.py` so one live frame maps the way the dataset did. `dataset_ranges()` recomputes them, and a test fails if the dataset drifts more than 2% from the frozen values. MediaPipe's x and y are fractions of the image width and height, so the hand is slightly stretched along x on a non-square camera. The frozen ranges absorb this because the dataset and the demo share the camera.

### 4.2 Pose library, built from your dataset

`build_poses.py`:
1. Load `landmarks_dataset.csv` and run `kinematics` on every row.
2. For each letter, take the **median** of each `HandPose` field across all sessions. The median is robust to bad frames.
3. Report per-letter spread (IQR). A wide IQR means signers were inconsistent or the letter is ambiguous for this hand.
4. **Shape** each letter with its ASL handshape (`hand/handshapes.py`, added in the naturalness pass). A finger that ASL extends becomes exactly 0 and one it closes exactly 1, and the index–middle spread becomes together (0) or apart (0.8). Only partial fingers (the curves of C, O, and D, E's bend, the fold of M and N, X's hook, T's index over the thumb) keep the measured curl. Two facts force this. MediaPipe reads a finger tucked under the thumb at a median 196° against a fist's 241°, so 42 of the 51 fingers that ASL closes came out 0.55–0.80 and would stop half-bent on the robot, sticking straight out from the palm. And the robot has one servo per finger, so "closed" is a single end stop, not a range. The report lists recordings that contradict a handshape outright (F, G, and L today) as letters to recollect; `test_handshapes` fails if a new one appears.
5. Apply `hand/pose_overrides.json` field by field. It holds hand-authored corrections with a `reason` string each. A value may be a letter name, which copies that letter's shaped value (`"thumb_opp": "L"`), so an override follows the data instead of freezing a number.
6. Add `J`, `Z` (motion stubs), `REST` (a relaxed hand: curl 0.12 → 0.26 from index to pinky, thumb relaxed, spread 0.2, the natural resting cascade), and `OPEN` (the flat, spread "5" hand).
7. Write `hand/poses.json`, sorted, with the dataset hash and date, so a stale table is detectable. `scripts/render_poses.py` draws every pose as a skeleton (`reports/poses_preview.png`) so naturalness can be checked by eye before any hardware exists.

The dataset already gives usable poses for most letters. Appendix A shows the first pass.

### 4.3 Mapping: HandPose → servo angles

`config/hand.yaml` is the only place hardware facts live:

```yaml
port: auto            # or /dev/tty.usbmodem1101, COM3
baud: 115200
rate_hz: 30
channels:             # PCA9685 channel → joint
  - {ch: 0, joint: thumb_flex, min_deg: 10, max_deg: 170, invert: false}
  - {ch: 1, joint: thumb_opp,  min_deg: 20, max_deg: 160, invert: false}
  - {ch: 2, joint: index,      min_deg: 0,  max_deg: 180, invert: false}
  - {ch: 3, joint: middle,     min_deg: 0,  max_deg: 180, invert: true}
  - {ch: 4, joint: ring,       min_deg: 0,  max_deg: 180, invert: false}
  - {ch: 5, joint: pinky,      min_deg: 0,  max_deg: 180, invert: true}
  - {ch: 6, joint: spread,     min_deg: 60, max_deg: 120, invert: false}
  - {ch: 7, joint: wrist_roll, min_deg: 0,  max_deg: 180, invert: false}
letter_hold_ms: 900   # minimum time a letter pose is held in spell mode
smoothing_alpha: 0.35 # EMA in mirror mode
```

`angle = min + value_01 × (max − min)`, inverted if `invert`, rounded, and clamped. The `min_deg`/`max_deg` values come from calibration (ROADMAP stage 4), where the user finds the angle at which each tendon is slack and the angle at which the finger is fully closed without stalling the servo.

### 4.4 Controller and threading

```
camera thread (main)                          hand thread
────────────────────                          ───────────
frame → landmarks → letter ──put_nowait──►  Queue(maxsize=1)  ──► mapper → link.send → wait OK (≤50 ms)
                     (drop old item if full)                        ↑ 30 Hz cap, EMA in mirror mode
```

- **Letter mode** triggers on the same "stable for 10 frames" event that already drives TTS (`app_opencv.py` around line 164). It sends only when the letter changes.
- **Mirror mode** sends every frame, capped at 30 Hz.
- If the link fails, the controller logs once, marks the HUD `HAND: offline`, and retries every 2 s. The demo keeps running.
- If no hand is detected for more than 1 s, it sends `REST`.

### 4.5 Integration into `app_opencv.py`

Three touch points only:
1. Add `argparse` flags: `--hand PORT|mock`, `--hand-mode letter|mirror` (default `letter`).
2. Build the controller if `--hand` is given, else `None`.
3. In the stable-prediction branch, call `controller.submit_letter(label)`. In mirror mode, call `controller.submit_landmarks(landmarks)` every frame. Add one HUD line showing hand status.

## 5. Mechanical hand

| Option | DOF actually driven | Looks human? | Cost | Verdict |
|---|---|---|---|---|
| **InMoov hand + forearm** (open-source, printed) | 5 finger tendons + wrist; add thumb swing | Yes, human proportions | ~$80–120 | **Recommended** |
| Flexy-Hand / e-NABLE remix | 5 tendons | Fairly | ~$40 | Fallback |
| Custom hand from the Blender model (`blender/`) | 8 | Yes | printing time | Later: the model is a visual/kinematic reference, not print-ready |

The ASL alphabet needs more than 5 servos. Why each extra channel exists:
- **Thumb swing (ch 1)**: A vs S vs T vs M vs N differ mainly in where the thumb sits. Your confusion matrix shows the same cluster.
- **Spread (ch 6)**: U vs V, and R. One servo drives a small linkage that abducts the index and middle fingers apart.
- **Wrist roll (ch 7)**: G/H (point sideways) and P/Q (point down). The InMoov wrist gear handles this.

For a human-looking finish (like the reference photo), cover the printed hand with a thin silicone glove or skin-tone TPU fingertip covers. This does not change the electronics.

## 6. Electronics and firmware

The full wiring is in `WIRING.md` and `wiring.svg`. In summary: **Arduino Uno + PCA9685 driver + separate 6 V / 10 A supply feeding a servo power bus**. Never power servos from the Arduino's 5 V pin: six MG996Rs can pull over 10 A combined when stalled, which resets the board or burns its regulator.

Firmware `firmware/collector_hand/collector_hand.ino` behaves as follows:
- Parses line commands (PROTOCOL.md) into a fixed 48-byte buffer.
- Keeps `target[8]` and `current[8]`. Every 20 ms it moves `current` toward `target` by at most `SLEW_DEG_PER_TICK` (default 6°, about 300°/s). This protects tendons and power.
- Compiled-in `HARD_MIN[8]` / `HARD_MAX[8]` limits act as a second clamp behind Python's.
- **Watchdog:** if no valid command arrives within 3 s, it moves to REST and then cuts PWM (`setPWM(ch, 0, 4096)` = full off) so servos stop buzzing.
- Optional E-stop button on D2 (INPUT_PULLUP, interrupt): cuts all PWM until `H` (home) is received.
- Status LED on D13: solid = connected, blinking = watchdog relaxed.
- Libraries: `Adafruit PWM Servo Driver Library` (+ `Wire`). Nothing else.

## 7. Error handling

| Failure | Detected by | Behavior |
|---|---|---|
| Arduino unplugged | `SerialException` / ack timeout | HUD "offline", retry every 2 s, recognizer unaffected |
| Garbled line | firmware replies `ERR 1` | Python logs it and resends the latest frame once |
| Python crashes | firmware watchdog (3 s) | hand relaxes, PWM off |
| Servo stall / brown-out | Arduino resets → sends `READY` on boot | Python re-sends current frame |
| Unknown letter (J/Z today) | `PoseLibrary` | falls back to the static part (I for J, index-point for Z) and logs it |

## 8. Testing strategy (no hardware needed)

- `tests/hand/test_kinematics.py` runs on the real dataset. The median `curl_01` for B and W (index, middle) < 0.15. A and S (all four) > 0.6. Y/I pinky < 0.15. L thumb_opp < U thumb_opp.
- `tests/hand/test_mapping.py`: clamping, inversion, and `min == max` edge cases, with properties checked over random poses.
- `tests/hand/test_protocol.py`: encode/decode round-trip, malformed lines rejected, max line length.
- `tests/hand/test_controller.py` with `MockLink`: latest-wins drops stale frames, rate cap holds, REST after timeout, reconnect.
- `tests/hand/test_poses_json.py`: `poses.json` covers all 26 letters plus REST/OPEN, and its dataset hash matches the CSV.
- `tests/hand/test_handshapes.py`: every letter pose matches its ASL handshape (closed fingers exactly 1, extended exactly 0, thumb and spread inside their state's range), C/O/E curve every finger alike, and no new recording contradicts ASL unnoticed.
- Firmware: `scripts/serial_selftest.py --port ...` (ping, sweep each channel 90→60→90, check `OK`s). This is the manual bench test in ROADMAP stage 3.

## 9. Findings from your dataset (read before building)

Found by running the kinematics sketch over `landmarks_dataset.csv`:

1. **Your F recordings look like B.** In all 20 F sessions the index finger is straight (median bend 12°, same as B's 14°), and the thumb–index distance is ~1.0 (open, not pinched). A real F pinches the index tip to the thumb, with a distance of ~0.2 (compare O at 0.13–0.33). This explains the documented F→B confusion better than "ASL ambiguity" does. The fix is to recollect F with a clear index–thumb circle. Until then, F comes from `pose_overrides.json`.
   *Confirmed in stage 2:* thumb-tip to index-tip distance is 0.95 for F (sessions 0.79–1.13) against 0.99 for B and 0.22 for O. The override copies O's index and thumb.
2. **Orientation letters (G, H, P, Q)** aren't captured, because the landmarks are normalized and the wrist angle carries through only weakly. Their finger curls are right, but the wrist roll/pitch must come from overrides.
   *Corrected in stage 2:* the conclusion holds, but the cause is different. `HandTracker` only translates and scales, so orientation survives normalization. The recordings themselves are upright: the wrist→9 direction for G, P, and Q is −5° to −7° from vertical, the same as every other letter. The exception is H, whose palm faces sideways (palm-normal roll +0.75 against a +0.15 baseline for all letters). The palm-normal roll in §4.1 detects that, and H's override keeps it. See §10.6 for what the hardware can show.
3. **U vs V vs R** have near-identical curls. Only the spread metric separates them, so the spread must use proximal phalanges (§4.1), not fingertips.
   *Confirmed in stage 2:* the proximal-phalanx angle separates every U session from every V session (U ≤ 5.2°, V ≥ 6.1°), but only after normalizing over frames with both fingers extended (§4.1). R reads between them (spread 0.27): its crossed fingers diverge, and the hand has no crossing DOF, so R looks like a slightly spread U.
4. **L's index reads 76°** because the MediaPipe `z` depth is noisy when the hand is turned sideways. Consider a 2D (x, y) fallback for the MCP angle when |z| variance is high. Stage 2 decides this.
   **Decision (stage 2): keep 3D; no 2D fallback.** The z-noise hypothesis does not hold. Medians over all frames of each letter:

   | Letter | MCP bend, 3D | MCP bend, 2D (x, y) | Index curl, 3D | Index curl, 2D |
   |---|---|---|---|---|
   | L | 97.3° | 99.7° | 119.2° | 121.1° |
   | G | 51.1° | 47.5° | 85.2° | 81.6° |
   | B (reference) | 7.0° | 5.1° | 15.0° | 10.2° |

   Dropping z changes L by 2°, so depth is not what bends it. In the image plane, L's index points 154° away from "up" with its tip below the wrist, while the wrist→index-MCP line points −48°: the recorded index is folded about 100° against the palm. Like F, this is a recording or tracking problem with this letter, not noise to filter. A 2D fallback isn't free either: dropping depth moves C's index curl from 159° to 177°, because C's fingers curve toward the camera and a projected angle distorts in either direction. L's index is straightened by its handshape (§4.2), which ASL defines as extended. Recollect L with the index pointing up and the palm to the camera.

## 10. Open decisions (ask the user)

1. Hand design: InMoov (recommended) or another design already owned?
2. Board: Uno (recommended; plenty for 8 channels) or ESP32 (adds Wi-Fi/BLE later, but runs at 3.3 V logic; the PCA9685 is fine with that)?
3. Is the spread servo (ch 6) worth the mechanical work, or should U/V/R be accepted as ambiguous in v1?
4. Should mirror mode ship in v1, or come after letter mode is solid?
5. **Where do the REST angles live?** (Found while writing the firmware model in stage 1; needed before stage 3.) `H` and the watchdog drive to REST inside the firmware, but "open hand" in servo degrees depends on the calibration and `invert` flags in `config/hand.yaml`: with today's defaults the index is open at 0° while the inverted middle finger is open at 180°. Options:
   - (a) Generate `firmware/collector_hand/hand_config.h` from `config/hand.yaml` plus the REST pose in `poses.json`, like the standalone table in PROTOCOL.md §5. This keeps one source of truth and needs no protocol change, but you reflash after each calibration. **Recommended.**
   - (b) Add a protocol command that sets REST at runtime, sent after every `READY`. No reflash, but it is a v2 protocol change, and the watchdog REST before Python connects is still a compiled-in guess.
   - (c) Define REST as "every horn at 90°" (the assembly neutral, WIRING.md §5) and let Python send the real open pose. Simplest, but the hand then goes to a half-closed pose on a watchdog relax.

   Until this is decided, `FirmwareSim` uses 90° on every channel.
6. **Orientation letters need more than wrist roll.** (Found while writing the stage-2 overrides.) ASL G and H point sideways and P and Q point down. Channel 7 rolls the hand about a vertical forearm (the Blender model and WIRING.md S7), which can turn the palm edge-on but cannot point the fingers sideways or down. The Blender pose table uses a 3-axis wrist for exactly these letters. In v1, `pose_overrides.json` turns G and H edge-on (roll +0.6) and keeps P and Q upright: P differs from K by its bent middle finger, and Q differs from G by facing the viewer. Options: (a) accept these approximations for v1; (b) add a wrist flexion servo as channel 8 (`ch=9`, a PROTOCOL v2 change, the PCA9685 has room); (c) mount the forearm on a tilting base. The robot-reads-robot test in stage 5 will measure whether (a) is good enough.

---

## Appendix A: first-pass servo angles from the dataset

*Superseded by `reports/poses_report.md`, which `scripts/build_poses.py` generates with the stage-2 normalization. Kept as the historical first pass: it used one shared curl range of 10–270°.*

Median per letter across all sessions, mapped linearly to 0–180° (0 = straight/open). This was produced by the §4.1 formulas before calibration, so it is for sanity-checking, not for flashing.

| Letter | Index | Middle | Ring | Pinky | Thumb flex | Thumb tip → pinky MCP | Note |
|---|---|---|---|---|---|---|---|
| A | 138° | 164° | 163° | 148° | 52° | 0.66 |  |
| B | 3° | 0° | 0° | 0° | 140° | 0.33 |  |
| C | 103° | 108° | 107° | 98° | 93° | 0.51 |  |
| D | 10° | 114° | 122° | 122° | 91° | 0.61 |  |
| E | 122° | 139° | 143° | 136° | 178° | 0.41 |  |
| F | 1° | 0° | 1° | 3° | 87° | 0.37 | looks like B (§9.1): override |
| G | 52° | 149° | 180° | 171° | 136° | 0.64 | orientation: override |
| H | 30° | 32° | 129° | 128° | 148° | 0.77 | orientation: override |
| I | 117° | 120° | 124° | 15° | 68° | 0.50 |  |
| K | 4° | 7° | 119° | 119° | 17° | 0.63 |  |
| L | 76° | 157° | 180° | 168° | 32° | 1.75 | index z-noise (§9.4) |
| M | 112° | 115° | 115° | 131° | 72° | 0.25 |  |
| N | 108° | 112° | 120° | 121° | 70° | 0.38 |  |
| O | 85° | 96° | 100° | 92° | 30° | 0.62 |  |
| P | 26° | 108° | 118° | 117° | 32° | 0.72 | orientation: override |
| Q | 7° | 115° | 124° | 123° | 16° | 0.84 | orientation: override |
| R | 7° | 5° | 116° | 113° | 84° | 0.38 |  |
| S | 144° | 180° | 180° | 179° | 66° | 0.45 |  |
| T | 119° | 157° | 173° | 175° | 25° | 0.50 |  |
| U | 5° | 5° | 119° | 122° | 92° | 0.31 |  |
| V | 3° | 4° | 119° | 120° | 79° | 0.30 |  |
| W | 1° | 0° | 4° | 101° | 121° | 0.47 |  |
| X | 56° | 128° | 156° | 160° | 96° | 0.46 |  |
| Y | 117° | 120° | 124° | 13° | 4° | 0.81 |  |
