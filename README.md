# Real-Time Sign Language Recognizer (Milestone 1)

A lightweight, high-performance sign language recognizer that runs locally on your CPU. This repository contains the complete end-to-end pipeline for **Milestone 1**: recognizing static letters **A, B, C, D, and E** using Google's MediaPipe hand-tracking and machine learning classifiers.

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

Run the collection script:
```bash
python data_collection.py
```
- Select which label you want to collect (`A`, `B`, `C`, `D`, or `E`).
- Position your hand in the screen when the countdown starts.
- Once completed, the landmarks and class labels are appended with a unique `session_id` into `landmarks_dataset.csv`.

---

### Stage 2: Offline Training & Evaluation
Train your classifiers and compare performance.
```bash
python train.py
```
This script will:
- Load `landmarks_dataset.csv`.
- Perform a **session-based split** (75% train / 25% test) to prevent data leakage and guarantee valid evaluation.
- Train a **Random Forest** (baseline) and a **PyTorch Neural Network** (improved model).
- Compare accuracies and output class metrics.
- Export `best_model.pth`, `random_forest.joblib`, and `label_map.joblib`.
- Generate a confusion matrix plot at `evaluation_confusion_matrix.png`.

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
*(Note: Requires browser camera permissions. Falls back to a desktop client recommendation if WebRTC is not supported on the host system.)*

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

