# Real-Time Sign Language Recognizer (Milestone 1)

A lightweight, high-performance sign language recognizer that runs locally on your CPU. This repository contains the complete end-to-end pipeline for **Milestone 1**: recognizing the **24 static letters** of the ASL fingerspelling alphabet using Google's MediaPipe hand-tracking and machine learning classifiers.

> **Scope — 24 of 26 letters.** `J` and `Z` are intentionally excluded: in ASL fingerspelling they are *motion* gestures (J traces a hook, Z draws a zigzag), so a single-frame static-pose classifier structurally cannot represent them. They are deferred to the Milestone-2 sequence model. All other letters (`A`–`Y` except `J`) are supported.

---

## System Architecture

The project consists of two main workflows:

```
[Offline Subsystem]
  Webcam Frame ──► MediaPipe (21 Points) ──► Normalization (Wrist & Scale) ──► Save to CSV (with session_id)
  CSV Dataset  ──► Session-Based Train/Test Split ──► Train RF & PyTorch NN ──► Export Best Model & Stats

[Real-Time Subsystem]
  Webcam Frame ──► MediaPipe (21 Points) ──► Normalization ──► Trained Model ──► Stable HUD prediction ──► TTS Audio Speak
```

---

## Installation & Setup

1. **Verify Python**: Ensure you have Python 3.9+ installed.
2. **Install Dependencies**: Run the following command from the project root directory:
   ```bash
   pip install -r requirements.txt
   ```

---

## Step-by-Step Workflow

### Stage 1: Data Collection
Collect your custom training dataset. To ensure robustness, you will record **8–10 short sessions** of 2-3 seconds each for each letter. 
Between sessions, you'll have 2 seconds to adjust your hand's angle, distance, and lighting.

Run the collection script for a single letter (interactive):
```bash
python data_collection.py
```
- Enter which letter to collect (`A`–`Z`, except `J`/`Z`, which are refused with an explanation).
- Position your hand in the screen when the countdown starts.
- Landmarks and labels are appended with a unique `session_id` into `landmarks_dataset.csv`. Re-running a letter auto-continues its session numbering (e.g. `F_11`), so recordings never collide.

Or collect several letters in one run (batch mode) — pass a letter string:
```bash
python data_collection.py MNST          # the four fist-cluster letters
python data_collection.py ABCDEFGHIJKLMNOPQRSTUVWXYZ   # whole alphabet
```
Batch mode walks each letter in turn, auto-skips `J`/`Z`, and offers to skip any letter that already has 10+ sessions — so passing the entire alphabet is safe.

---

### Stage 2: Offline Training & Evaluation
Train your classifiers and compare performance.
```bash
python train.py
```
This script will:
- Load `landmarks_dataset.csv`.
- Perform a **session-based train / validation / test split** (~70/15/15, whole sessions held out) to prevent data leakage. The validation set selects the best epoch checkpoint *and* the winning model; the test set is evaluated exactly once, so the reported accuracy is not inflated by selecting on the data it reports.
- Train a **Random Forest** (baseline) and a **PyTorch Neural Network** (improved model).
- Export `best_model.pth`, `random_forest.joblib`, and `label_map.joblib`.
- Generate a confusion matrix plot at `evaluation_confusion_matrix.png` (figure size scales with the class count).

### Results (24-letter model)

| Model | Val accuracy | Test accuracy |
|-------|-------------|---------------|
| Random Forest (baseline) | 83.5% | 85.7% |
| **PyTorch NN (winning)** | **85.4%** | **87.9%** |

Test accuracy is on a held-out set of sessions the model never trained or validated on. Strongest letters (F1 ≈ 1.0): `D`, `M`, `N`, `O`, `Q`. Known confusion pairs — all consistent with genuine ASL fingerspelling ambiguity — are `F↔B`, `L↔G`, `U↔V`, and the `A`/`S`/`T` closed-fist cluster (they differ mainly by thumb position). These are documented rather than hidden; `F→B` is the largest single error and the priority for further data collection.

---

### Stage 3: Real-Time Prediction Live Demo

#### Option A: High-Performance OpenCV Desktop Client (Recommended)
This runs directly on your desktop window with smooth FPS and thread-safe text-to-speech audio feedback.
```bash
python app_opencv.py
```
- Select between the baseline Random Forest or PyTorch Neural Network.
- Press `q` to close the webcam stream.

#### Option B: Streamlit Web Client
This opens a dashboard in your browser with sidebar controls.
```bash
streamlit run app_streamlit.py
```
*(Note: Requires browser camera permissions. Falls back to a desktop client recommendation if WebRTC is not supported on the host system. The OpenCV desktop client is the recommended demo surface — it is lower-latency and does not depend on WebRTC.)*

---

## Mathematical Normalization Details
To ensure the model is invariant to where your hand is and how far it is from the camera:
1. **Translation**: All 21 landmarks $(x, y, z)$ are translated relative to the wrist (landmark 0), placing the wrist at $(0,0,0)$.
2. **Scaling**: Coordinates are divided by the Euclidean distance between the wrist (landmark 0) and the middle-finger knuckle (landmark 9).

---

## Recent Updates & Completed Work
We have successfully updated the codebase to support modern MediaPipe versions (v0.10.x and newer):

1. **MediaPipe Tasks API Migration**:
   - Updated `utils/hand_tracker.py` to use the new `mediapipe.tasks.python.vision.HandLandmarker` instead of the legacy `mp.solutions.hands` API which is deprecated/removed in modern versions.
   - Tied it to the pre-existing `hand_landmarker.task` model file.

2. **Custom Hand Landmark Drawing**:
   - Legacy MediaPipe drawing utilities (`mp.solutions.drawing_utils` and `mp.solutions.drawing_styles`) are deprecated.
   - Implemented a lightweight, custom `_draw_hand_landmarks` drawing helper in `utils/hand_tracker.py` that utilizes OpenCV lines and circles to render the hand joints and connections.

3. **Compatibility Improvements**:
   - Fixed startup imports in `app_opencv.py` and `utils/hand_tracker.py` to prevent `ModuleNotFoundError: No module named 'mediapipe.framework'`.
   - Cleaned up MediaPipe resources on exit using `tracker.close()`.

