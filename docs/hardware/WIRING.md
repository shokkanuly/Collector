# Collector Hand: Wiring and Arduino Details

Diagram: `wiring.svg` / `wiring.png` (same folder). This page is the text version: every wire, and where it goes.

> **Safety first.** Servos run from their own 6 V supply, never from the Arduino. All grounds are joined. Build and test with **one servo and no tendons** before connecting the whole hand (ROADMAP stage 3).

## 1. Parts list (BOM)

Prices are rough estimates. Check your local stores (in Kazakhstan: Chip&Dip, Kaspi, or AliExpress).

| # | Part | Qty | Notes | ≈ USD |
|---|---|---|---|---|
| 1 | Arduino Uno R3 (or clone with CH340) | 1 | ATmega328P, 16 MHz, 5 V logic | 8–25 |
| 2 | PCA9685 16-ch 12-bit PWM servo driver | 1 | I²C address 0x40 (default) | 4–10 |
| 3 | MG996R metal-gear servo | 6 | thumb flex, 4 fingers, wrist. 4.8–7.2 V, ~10 kg·cm, stall ≈ 2.5 A | 25–35 |
| 4 | MG90S metal-gear micro servo | 2 | thumb swing, finger spread. 4.8–6 V | 5–8 |
| 5 | 6 V DC power supply, **10 A** (60 W) | 1 | or a 5 V/10 A supply with trim pot set to 6.0 V (e.g. Mean Well LRS-75-5) | 12–20 |
| 6 | Blade fuse holder + 10 A fuse | 1 | on the + line right after the supply | 2 |
| 7 | Rocker switch, rated ≥ 10 A | 1 | main servo power | 1 |
| 8 | Electrolytic capacitor 1000 µF, ≥ 16 V | 1 | across the servo bus, absorbs current spikes | 0.5 |
| 9 | Terminal block / Wago 221 connectors or a strip board | 2 | makes the +6 V and GND bus | 3 |
| 10 | Momentary push button (normally open) | 1 | E-stop on D2 | 1 |
| 11 | Wire: 16–18 AWG red/black (power), 22 AWG Dupont jumpers (logic) | – | | 5 |
| 12 | Servo extension cables 30 cm | 8 | servos sit in the forearm | 4 |
| 13 | Braided fishing line 0.5 mm (40–80 lb) + 1 mm elastic cord | – | tendons and return springs | 5 |
| 14 | PLA filament, M3 screws, bearings (per InMoov hand BOM) | – | printed hand + forearm | 20–30 |
| 15 | USB-A to USB-B cable | 1 | laptop ↔ Arduino | 2 |

Total ≈ **$100–160**.

## 2. Connections

### 2.1 Laptop ↔ Arduino
| From | To | Wire |
|---|---|---|
| Laptop USB | Arduino USB-B | USB cable. Carries serial data **and** powers the Arduino logic |

### 2.2 Arduino ↔ PCA9685 (logic, 4 wires)
| Arduino Uno pin | PCA9685 pin | Color (suggested) | Purpose |
|---|---|---|---|
| **5V** | **VCC** | pink/red | powers the PCA9685 chip only (logic, a few mA) |
| **GND** | **GND** | black | logic ground |
| **A4** (SDA) | **SDA** | blue | I²C data |
| **A5** (SCL) | **SCL** | green | I²C clock |
| – | OE | – | leave unconnected (the board pulls it low = outputs on) |
| – | V+ (header pin) | – | leave unconnected |

On Uno R3 you can also use the dedicated SDA/SCL pins near AREF. They are the same signals as A4/A5.

### 2.3 Power (high current)
| From | To | Wire |
|---|---|---|
| PSU **+6 V** | Switch → 10 A fuse → **+6 V bus** | 16–18 AWG red |
| PSU **−** | **GND bus** | 16–18 AWG black |
| 1000 µF capacitor **+** (long leg) | +6 V bus | short leads, **observe polarity** |
| 1000 µF capacitor **−** (striped side) | GND bus | |
| **GND bus** | PCA9685 screw terminal **GND** | 18–20 AWG black: this is the **common ground** that joins servo power and logic |
| PCA9685 screw terminal **V+** | – | leave empty: servos take power from the bus, not through the PCA9685 board |

Why a bus instead of the PCA9685 V+ terminal? The board's thin traces are fine for micro servos, but six MG996Rs under load can draw 5–10 A. A bus keeps that current off the PCB.

### 2.4 Servos (3 wires each)
Every servo cable has: **brown/black = GND**, **red = +6 V**, **orange/yellow = signal**.

| Servo | Joint | Model | Signal → PCA9685 | Red → | Brown → |
|---|---|---|---|---|---|
| S0 | thumb flex | MG996R | CH0 (PWM pin) | +6 V bus | GND bus |
| S1 | thumb swing (opposition) | MG90S | CH1 | +6 V bus | GND bus |
| S2 | index | MG996R | CH2 | +6 V bus | GND bus |
| S3 | middle | MG996R | CH3 | +6 V bus | GND bus |
| S4 | ring | MG996R | CH4 | +6 V bus | GND bus |
| S5 | pinky | MG996R | CH5 | +6 V bus | GND bus |
| S6 | index–middle spread | MG90S | CH6 | +6 V bus | GND bus |
| S7 | wrist rotation | MG996R | CH7 | +6 V bus | GND bus |

Simple option: plug each servo's 3-pin connector straight into the PCA9685 header (PWM / V+ / GND row). Then cut or pull out only the **red** pin of each connector and wire it to the +6 V bus instead. The ground still reaches the bus through the PCA9685 GND terminal.

### 2.5 E-stop and status
| Arduino pin | Connects to | Notes |
|---|---|---|
| **D2** | push button, one leg | `pinMode(2, INPUT_PULLUP)`; interrupt on FALLING |
| **GND** | push button, other leg | |
| D13 | on-board LED | solid = connected, blinking = watchdog relaxed |

## 3. Arduino details

| Item | Value |
|---|---|
| Board | Arduino Uno R3, FQBN `arduino:avr:uno` |
| MCU | ATmega328P, 16 MHz, 32 KB flash, **2 KB RAM** (use fixed `char` buffers, not `String`) |
| Serial | USB CDC, 115200 baud, 8N1 (protocol: `PROTOCOL.md`) |
| I²C | Wire library, 400 kHz (`Wire.setClock(400000)`); PCA9685 at 0x40 |
| Library | **Adafruit PWM Servo Driver Library** (installs Adafruit BusIO) |
| PWM frequency | 50 Hz (`pwm.setPWMFreq(50)`); call `pwm.setOscillatorFrequency(27000000)` and trim ±1 MHz if angles drift |
| Pulse → counts | 1 count = 20 ms / 4096 ≈ 4.88 µs. Start with **600 µs → 123 counts** (0°) and **2400 µs → 492 counts** (180°). Refine per servo in calibration |
| Servo tick | every 20 ms, slew-limited (default 6°/tick) |
| Watchdog | 3 s without a command → REST → PWM off |
| Pins used | A4, A5 (I²C), D2 (E-stop), D13 (LED), USB. All others are free for sensors later (e.g. FSR fingertip pressure on A0–A3) |

Install and build:
```bash
arduino-cli core install arduino:avr
arduino-cli lib install "Adafruit PWM Servo Driver Library"
arduino-cli compile --fqbn arduino:avr:uno firmware/collector_hand
arduino-cli upload  --fqbn arduino:avr:uno -p /dev/tty.usbmodem1101 firmware/collector_hand
```
On macOS the port looks like `/dev/tty.usbmodemXXXX` (genuine Uno) or `/dev/tty.wchusbserialXXXX` (CH340 clones, which may need the CH340 driver).

## 4. Power budget

| Load | Typical moving | Stall (worst case) |
|---|---|---|
| 6 × MG996R @ 6 V | 6 × 0.5–0.9 A | 6 × 2.5 A = 15 A |
| 2 × MG90S @ 6 V | 2 × 0.2 A | 2 × 0.7 A |
| **Total** | **≈ 3.5–6 A** | **≈ 16 A** |

The 10 A supply covers normal use. Three things keep the hand out of the worst case: firmware slew limiting (not all servos start at once), calibration that stops each finger **before** it hits its end stop (a pressing servo is a stalled servo), and the watchdog relax. If a servo buzzes while holding a pose, reduce that channel's `max_deg` by 5°.

## 5. Mechanical: tendons to servos

1. Power the servos, send `C <ch> 90` to each, then fit the horns centered. Use a **double-arm horn**.
2. Each finger has two lines. The **flexor** (palm side) goes to one horn arm, and the **extensor** (back side, or elastic cord) goes to the opposite arm, so turning the horn pulls one line and releases the other.
3. Tie the lines with the finger straight and the servo at the "open" end of its range. Pre-tension them lightly with the InMoov tensioner screws.
4. Route lines through PTFE tube in the wrist so wrist rotation doesn't pull the fingers.
5. Then run `hand_cli.py calibrate` (ROADMAP stage 4).

## 6. First power-on checklist

- [ ] Multimeter: PSU reads 5.9–6.2 V **before** connecting servos.
- [ ] No continuity between +6 V bus and GND bus.
- [ ] Continuity between Arduino GND and GND bus (common ground).
- [ ] Capacitor polarity correct.
- [ ] Arduino shows `READY v1 ch=8` in the serial monitor at 115200.
- [ ] `P` → `PONG`; `C 0 90` moves only S0; E-stop cuts motion; 3 s silence → `WDT`.
