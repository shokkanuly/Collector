# Hardware Roadmap: Collector Hand

Same rules as the main `ROADMAP.md`: do one stage at a time, commit after each step, and "done when" is the gate. Stages 0–2 need no hardware, so they can start today.

---

## Stage 0: Scaffolding (no hardware)

1. Create `hand/` with `types.py`, empty modules, and `__init__.py`.
2. Add `requirements-hand.txt` (`pyserial>=3.5`, `pyyaml>=6`) and `config/hand.yaml` (ARCHITECTURE §4.3 defaults).
3. Create `tests/hand/`, and make `python -m pytest tests/hand` run (one smoke test).

**Done when:** `python test_pipeline.py` and `pytest tests/hand` are both green, and `import hand` doesn't import serial.

## Stage 1: Protocol + MockLink (no hardware)

1. `hand/protocol.py`: encode/decode for every command and reply in PROTOCOL.md.
2. `hand/link.py`: `Link` interface, `MockLink` (records frames, simulates `OK`/`ERR`/`WDT`), and `SerialLink` (lazy pyserial, waits for `READY`, ack timeout, reconnect).
3. Tests: round-trips, malformed input, line-length limit, timeout path.

**Done when:** 100% of PROTOCOL.md commands have a passing encode test and a MockLink test.

## Stage 2: Kinematics + pose library from the dataset (no hardware)

1. `hand/kinematics.py` (ARCHITECTURE §4.1), vectorized over `(N,63)`.
2. `scripts/build_poses.py` → `hand/poses.json` and `reports/poses_report.md` (per-letter median + IQR table).
3. `hand/pose_overrides.json` for F, G, H, P, Q (and L if §9.4 holds), each with a `reason`.
4. Decide the MCP z-noise question (§9.4): compare 3D vs 2D angles on L and G, and record the choice in ARCHITECTURE.md.
5. Tests from ARCHITECTURE §8.

**Done when:** `poses.json` has all 24 static letters plus J, Z, REST, and OPEN. The report shows B/W/U/V index curl < 0.15 and A/S/T > 0.6, and U vs V differ in `spread` by at least 0.3.

## Stage 3: Firmware + bench test (hardware: Arduino + PCA9685 + 1 servo)

1. `firmware/collector_hand/collector_hand.ino` per ARCHITECTURE §6 and PROTOCOL.md.
2. `scripts/serial_selftest.py`.
3. Bench-test on **one servo, no hand attached**: wire it per WIRING.md, then run ping, sweep, watchdog, E-stop, and over-long line tests.

**Done when:** the self-test passes against the real board and `arduino-cli compile` is clean with no warnings.

## Stage 4: Assembly + calibration (hardware: full hand)

1. Print and assemble the hand, then thread tendons (see WIRING.md §5 for tendon routing to servo horns).
2. Mount each servo at 90° with its horn centered **before** tying tendons.
3. `hand_cli.py calibrate`: steps each channel interactively (arrow keys ±2°). The user marks "slack" and "closed", and the values are written to `config/hand.yaml`.
4. `hand_cli.py spell ABCDEFGHIKLMNOPQRSTUVWXY` and record a video.

**Done when:** every letter is recognizable to a human and no servo stalls (buzzing or heat) while holding a pose for 5 s.

## Stage 5: Close the loop with the recognizer

1. Add `HandController` + `app_opencv.py` flags (ARCHITECTURE §4.4–4.5).
2. **Robot-reads-robot test:** point the webcam at the robotic hand, run `spell`, and log what the NN predicts for each letter. The result is a confusion table in `reports/robot_readback.md`. It's an honest end-to-end metric and a good MIT-application story.

**Done when:** the live demo drives the hand with no camera FPS drop (±2 FPS), and the read-back accuracy is reported.

## Stage 6: Mirror mode (optional for v1)

EMA smoothing, a 30 Hz cap, and REST when the hand is lost. Measure end-to-end latency (camera → servo) with a phone slow-mo video.

**Done when:** the latency is measured and documented, and the hand tracks an open/close motion without oscillating.
