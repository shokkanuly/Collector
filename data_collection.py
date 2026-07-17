import cv2
import time
import os
import csv
import numpy as np
import pandas as pd
from utils.hand_tracker import HandTracker, _draw_hand_landmarks

def collect_data():
    print("==================================================")
    print("        Sign Language Data Collection Module      ")
    print("==================================================")
    
    # Get class label from user
    while True:
        label = input("Enter static sign label to collect (A-E): ").strip().upper()
        if label in ['A', 'B', 'C', 'D', 'E']:
            break
        print("Invalid label. Please enter a letter between A and E.")
    
    num_sessions = 10
    samples_per_session = 15  # At 5 FPS, this is 3 seconds of capture
    frame_interval = 0.2  # 5 FPS
    
    csv_path = 'landmarks_dataset.csv'

    # Continue numbering after any sessions already recorded for this label, so that
    # re-running for the same letter appends new sessions instead of colliding with
    # the existing ones and silently merging two recordings under one session_id.
    session_offset = 0
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        prior = existing.loc[existing['label'] == label, 'session_id'].unique()
        if len(prior) > 0:
            session_offset = max(int(s.split('_')[1]) for s in prior)
            print(f"\nFound {len(prior)} existing session(s) for '{label}'.")
            print(f"New sessions will be numbered from {label}_{session_offset + 1:02d}.")

    # Initialize tracker and webcam
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam. Please check your camera connection and permissions.")
        return

    print(f"\nReady to collect data for label: {label}")
    print(f"You will record {num_sessions} short sessions.")
    print("For each session, vary your hand angle, distance, and lighting.")
    print("Press 'q' at any time to exit the script.")
    input("\nPress Enter to start...")

    for session_idx in range(1, num_sessions + 1):
        session_id = f"{label}_{session_offset + session_idx:02d}"
        print(f"\n--- Starting Session {session_idx}/{num_sessions} (ID: {session_id}) ---")
        
        # 1. Countdown Phase (2 seconds)
        countdown_start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1) # Flip horizontally for natural mirror view
            h, w, _ = frame.shape
            
            elapsed = time.time() - countdown_start
            time_left = 2.0 - elapsed
            
            # Display countdown text
            display = frame.copy()
            if time_left > 0:
                cv2.putText(display, f"GET READY FOR SESSION {session_idx}/{num_sessions}", 
                            (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, f"Adjust hand pose & lighting", 
                            (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(display, f"{int(time_left) + 1}", 
                            (w // 2 - 30, h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 165, 255), 6)
            else:
                break
                
            cv2.imshow('Sign Language Data Collection', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Data collection aborted by user.")
                cap.release()
                cv2.destroyAllWindows()
                return

        # 2. Recording Phase (15 samples)
        recorded_samples = 0
        last_capture_time = 0
        
        while recorded_samples < samples_per_session:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            
            current_time = time.time()
            landmarks, hand_lms = tracker.extract_landmarks(frame)
            
            # Check if it is time to capture a sample (5 FPS)
            should_save = False
            if current_time - last_capture_time >= frame_interval:
                if landmarks is not None:
                    # Save landmark data
                    file_exists = os.path.exists(csv_path)
                    with open(csv_path, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            # 21 landmarks * 3 coordinates (x, y, z) + label + session_id
                            header = []
                            for i in range(21):
                                header.extend([f"l{i}_x", f"l{i}_y", f"l{i}_z"])
                            header.extend(["label", "session_id"])
                            writer.writerow(header)
                        
                        row = list(landmarks) + [label, session_id]
                        writer.writerow(row)
                        
                    recorded_samples += 1
                    last_capture_time = current_time
                    should_save = True

            # Visual overlay
            if hand_lms is not None:
                _draw_hand_landmarks(display, hand_lms)
                
            # Recording Status text
            cv2.putText(display, f"RECORDING SESSION {session_idx}/{num_sessions}", 
                        (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display, f"Sample: {recorded_samples}/{samples_per_session}", 
                        (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show a green flash on-screen when a sample is successfully saved
            if should_save:
                cv2.rectangle(display, (0, 0), (w, h), (0, 255, 0), 10)

            cv2.imshow('Sign Language Data Collection', display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Data collection aborted by user.")
                cap.release()
                cv2.destroyAllWindows()
                return

        # 3. Inter-session Pause / Adjust Phase (except for last session)
        if session_idx < num_sessions:
            pause_start = time.time()
            while time.time() - pause_start < 2.0:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                display = frame.copy()
                
                # Instruct user to adjust hand position
                cv2.putText(display, "PAUSED - ADJUST HAND POSE", 
                            (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)
                cv2.putText(display, "Move hand angle, distance, or lighting", 
                            (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(display, "Resuming shortly...", 
                            (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                cv2.imshow('Sign Language Data Collection', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Data collection aborted by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    print("\n==================================================")
    print("      Data Collection Completed Successfully!    ")
    print(f"Data saved/appended to: {os.path.abspath(csv_path)}")
    print("==================================================")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    collect_data()
