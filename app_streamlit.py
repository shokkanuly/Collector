import streamlit as st
import cv2
import numpy as np
import os
import joblib
import queue
import threading
import pyttsx3
from utils.hand_tracker import HandTracker, _draw_hand_landmarks

# Try to import WebRTC components
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

# PyTorch Imports
try:
    import torch
    import torch.nn as nn
    
    class FeedForwardNN(nn.Module):
        def __init__(self, input_dim=63, output_dim=5):
            super(FeedForwardNN, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, output_dim)
            )
            
        def forward(self, x):
            return self.network(x)
except ImportError:
    torch = None

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="Sign Language Recognizer",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Mode & Glassmorphism Styling
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00f2fe;
    }
</style>
""", unsafe_allow_html=True)

# 2. Asynchronous TTS Queue Setup
speech_queue = queue.Queue()
tts_active = True

def speech_worker():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 165)
        while True:
            text = speech_queue.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()
            speech_queue.task_done()
    except Exception as e:
        pass

# Start speaker daemon
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

# Load models and configurations
@st.cache_resource
def load_resources():
    has_rf = os.path.exists('random_forest.joblib')
    has_nn = os.path.exists('best_model.pth')
    has_labels = os.path.exists('label_map.joblib')
    
    classes = joblib.load('label_map.joblib') if has_labels else ['A', 'B', 'C', 'D', 'E']
    
    rf_model = joblib.load('random_forest.joblib') if has_rf else None
    
    nn_model = None
    if has_nn and torch is not None:
        device = torch.device('cpu')  # Force cpu for streamlit hosting
        nn_model = FeedForwardNN(input_dim=63, output_dim=len(classes))
        nn_model.load_state_dict(torch.load('best_model.pth', map_location=device))
        nn_model.eval()
        
    return classes, rf_model, nn_model

classes, rf_model, nn_model = load_resources()

# Sidebar Settings
st.sidebar.markdown("### ⚙️ Engine Settings")
model_option = st.sidebar.selectbox(
    "Classifier Architecture",
    ["Random Forest (Baseline)", "PyTorch Neural Net (Improved)"] if (rf_model and nn_model) else 
    ["Random Forest (Baseline)"] if rf_model else 
    ["PyTorch Neural Net (Improved)"] if nn_model else ["No models trained"]
)

enable_tts = st.sidebar.checkbox("Voice Output (Text-to-Speech)", value=True)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 1.0, 0.75, 0.05)

# Primary Layout
st.markdown("<h1 class='main-header'>Real-Time Sign Language Recognizer</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Milestone 1 Prototype: Static Letters A through E using MediaPipe Hand Landmarks.</p>", unsafe_allow_html=True)

col_demo, col_stats = st.columns([2, 1])

with col_stats:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 System Information")
    st.write(f"**Loaded Classes:** `{classes}`")
    st.write(f"**Normalization Mode:** Wrist-relative + Scale Normalized")
    
    if os.path.exists('landmarks_dataset.csv'):
        import pandas as pd
        df = pd.read_csv('landmarks_dataset.csv')
        st.write(f"**Dataset Rows:** `{len(df)}`")
        st.write(f"**Recorded Sessions:** `{df['session_id'].nunique()}`")
    else:
        st.info("No offline training dataset found locally.")
        
    st.markdown("</div>", unsafe_allow_html=True)

    # Show Confusion Matrix plot if exists
    if os.path.exists('evaluation_confusion_matrix.png'):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Evaluation Matrix")
        st.image('evaluation_confusion_matrix.png', caption='Session-Held-Out Test Results', use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col_demo:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🎥 Live Recognition Video Stream")
    
    # Check if models are available
    if not rf_model and not nn_model:
        st.warning("⚠️ No trained models detected. Please complete the collection and training stages first.")
    else:
        # Define WebRTC Video Processor
        class SignLanguageProcessor(VideoProcessorBase):
            def __init__(self):
                self.tracker = HandTracker()
                # Local copy of classes/models for the thread
                self.classes = classes
                self.rf_model = rf_model
                self.nn_model = nn_model
                self.model_choice = model_option
                self.confidence_threshold = confidence_threshold
                self.enable_tts = enable_tts
                
                # Stability variables
                self.last_prediction = None
                self.prediction_counter = 0
                self.last_spoken = None

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img = cv2.flip(img, 1) # Mirror flip
                h, w, _ = img.shape
                
                landmarks, hand_lms = self.tracker.extract_landmarks(img)
                
                predicted_label = None
                confidence = 0.0
                
                if landmarks is not None:
                    # Run inference
                    if "Random Forest" in self.model_choice and self.rf_model is not None:
                        probs = self.rf_model.predict_proba([landmarks])[0]
                        pred_idx = np.argmax(probs)
                        predicted_label = self.classes[pred_idx]
                        confidence = probs[pred_idx]
                    elif self.nn_model is not None and torch is not None:
                        with torch.no_grad():
                            inputs = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
                            outputs = self.nn_model(inputs)
                            probs = torch.softmax(outputs, dim=1).numpy()[0]
                            pred_idx = np.argmax(probs)
                            predicted_label = self.classes[pred_idx]
                            confidence = probs[pred_idx]
                            
                    # Draw visual skeleton
                    _draw_hand_landmarks(img, hand_lms)
                    
                    # Stable prediction speech trigger
                    if confidence >= self.confidence_threshold:
                        if predicted_label == self.last_prediction:
                            self.prediction_counter += 1
                            if self.prediction_counter >= 15: # Stable for 15 frames
                                if predicted_label != self.last_spoken and self.enable_tts:
                                    speech_queue.put(predicted_label)
                                    self.last_spoken = predicted_label
                        else:
                            self.last_prediction = predicted_label
                            self.prediction_counter = 0
                else:
                    self.prediction_counter = 0
                    self.last_prediction = None
                    
                # Visual HUD overlays
                overlay = img.copy()
                cv2.rectangle(overlay, (20, 20), (w - 20, 110), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
                
                # Render HUD text
                cv2.putText(img, f"SIGN RECOGNIZER (STREAMLIT)", (40, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                if predicted_label is not None and confidence >= self.confidence_threshold:
                    cv2.putText(img, f"LETTER: {predicted_label} ({int(confidence*100)}%)", (40, 95), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                else:
                    cv2.putText(img, "ALIGN HAND GESTURE IN CAMERA FRAME", (40, 95), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Stream Video
        if HAS_WEBRTC:
            ctx = webrtc_streamer(
                key="sign-recognizer",
                video_processor_factory=SignLanguageProcessor,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
            
            # Dynamically push parameter updates to running processor
            if ctx.video_processor:
                ctx.video_processor.model_choice = model_option
                ctx.video_processor.confidence_threshold = confidence_threshold
                ctx.video_processor.enable_tts = enable_tts
        else:
            st.info("Browser WebRTC backend not available or not installed. Falling back to local video loop description.")
            st.write("For complete real-time video inside Streamlit, install `streamlit-webrtc` and `av`.")
            st.write("Meanwhile, you can run the high-performance local viewer with:")
            st.code("python app_opencv.py")
            
    st.markdown("</div>", unsafe_allow_html=True)

# Instructions card
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📖 Setup & Instructions")
st.markdown("""
1. **Collect Data**: Run `python data_collection.py` from the command line and demonstrate static hand shapes for A, B, C, D, E.
2. **Train Models**: Run `python train.py` to process the landmarks and fit both classifiers.
3. **Run Demo**: Select your model and threshold in the sidebar. Show a gesture, and the system will output both a visual confirmation and a voice response.
""")
st.markdown("</div>", unsafe_allow_html=True)
