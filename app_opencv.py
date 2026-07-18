import cv2
import time
import os
import queue
import threading
import numpy as np
import joblib

# PyTorch Imports (only loaded if PyTorch model is used)
try:
    import torch
    from utils.model import FeedForwardNN
except ImportError:
    torch = None

import pyttsx3
from utils.hand_tracker import HandTracker

# 1. Thread-safe Speech Queue Worker
speech_queue = queue.Queue()
tts_enabled = True

def speech_worker():
    global tts_enabled
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)  # Moderate reading pace
        engine.setProperty('volume', 1.0)
        while True:
            text = speech_queue.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()
            speech_queue.task_done()
    except Exception as e:
        print(f"Warning: TTS initialization failed ({e}). Running in visual-only mode.")
        tts_enabled = False

# Start speech thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def main():
    print("==================================================")
    print("        Sign Language Real-time OpenCV Demo       ")
    print("==================================================")
    
    # Load Label Mapping
    if not os.path.exists('label_map.joblib'):
        print("Error: Label map 'label_map.joblib' not found. Run 'train.py' first.")
        return
    classes = joblib.load('label_map.joblib')
    print(f"Loaded classes: {classes}")
    
    # Check for available models
    has_rf = os.path.exists('random_forest.joblib')
    has_nn = os.path.exists('best_model.pth')
    
    if not (has_rf or has_nn):
        print("Error: No trained models found. Train models first using 'train.py'.")
        return
        
    # Model Selection UI (Console-based prompt)
    print("\nSelect Inference Model:")
    if has_rf:
        print("[1] Random Forest (Baseline)")
    if has_nn and torch is not None:
        print("[2] PyTorch Neural Network (Improved)")
        
    choice = "1"
    if has_rf and has_nn and torch is not None:
        while True:
            choice = input("Enter choice (1 or 2): ").strip()
            if choice in ['1', '2']:
                break
            print("Invalid input.")
    elif has_nn and torch is not None:
        choice = "2"
        print("Automatically using PyTorch model (Random Forest not found).")
    else:
        choice = "1"
        print("Automatically using Random Forest model (PyTorch or weights not found).")
        
    # Load Chosen Model
    model_type = "Random Forest"
    rf_model = None
    nn_model = None
    device = None
    
    if choice == "1":
        print("Loading Random Forest model...")
        rf_model = joblib.load('random_forest.joblib')
    else:
        print("Loading PyTorch model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nn_model = FeedForwardNN(input_dim=63, output_dim=len(classes))
        nn_model.load_state_dict(torch.load('best_model.pth', map_location=device))
        nn_model.to(device)
        nn_model.eval()
        model_type = "Neural Network"
        
    print(f"Successfully loaded {model_type}!")
    
    # Initialize Camera & Tracker
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return
        
    from utils.hand_tracker import _draw_hand_landmarks
    
    # Stability Tracking Variables
    last_prediction = None
    prediction_counter = 0
    last_spoken = None
    no_hand_counter = 0
    
    print("\nStarting live recognition screen.")
    print("Hold your hand in front of the camera. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip horizontally for mirrored view
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Create visual canvas for overlays
        display = frame.copy()
        
        # Extract features
        landmarks, hand_lms = tracker.extract_landmarks(frame)
        
        predicted_label = None
        confidence = 0.0
        
        if landmarks is not None:
            no_hand_counter = 0
            
            # Predict using selected model
            if choice == "1":  # Random Forest
                probs = rf_model.predict_proba([landmarks])[0]
                pred_idx = np.argmax(probs)
                predicted_label = classes[pred_idx]
                confidence = probs[pred_idx]
            else:  # Neural Network
                with torch.no_grad():
                    inputs = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(device)
                    outputs = nn_model(inputs)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                    pred_idx = np.argmax(probs)
                    predicted_label = classes[pred_idx]
                    confidence = probs[pred_idx]
            
            # Draw skeletons
            _draw_hand_landmarks(display, hand_lms)
            
            # Stability tracking logic
            if predicted_label == last_prediction:
                prediction_counter += 1
                if prediction_counter >= 10:  # Must be stable for 10 frames (~0.3 seconds)
                    if predicted_label != last_spoken and tts_enabled:
                        speech_queue.put(predicted_label)
                        last_spoken = predicted_label
            else:
                last_prediction = predicted_label
                prediction_counter = 0
                
        else:
            # No hand detected
            no_hand_counter += 1
            if no_hand_counter >= 10:
                last_prediction = None
                prediction_counter = 0
                last_spoken = None  # Reset speaking state
                
        # Premium HUD Design (Overlay)
        # Semi-transparent header card
        overlay = display.copy()
        cv2.rectangle(overlay, (20, 20), (w - 20, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
        
        # Display titles
        cv2.putText(display, f"SIGN RECOGNIZER ({model_type})", (40, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if predicted_label is not None:
            # Large letter box
            cv2.putText(display, f"LETTER: {predicted_label}", (40, 95), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # Draw confidence progress bar
            bar_start = 280
            bar_length = int(200 * confidence)
            cv2.putText(display, f"CONF: {int(confidence * 100)}%", (bar_start - 80, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.rectangle(display, (bar_start, 78), (bar_start + 200, 92), (50, 50, 50), -1)
            cv2.rectangle(display, (bar_start, 78), (bar_start + bar_length, 92), (0, 255, 0), -1)
        else:
            cv2.putText(display, "SHOW HAND SIGN IN FRAME", (40, 95), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        
        cv2.imshow('Sign Language Recognizer', display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    
    # Shutdown speech queue
    speech_queue.put(None)

if __name__ == "__main__":
    main()
