import cv2
import numpy as np
import os

# MediaPipe Tasks API (v0.10+)
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
# Path to the bundled hand landmarker model
_TASK_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hand_landmarker.task")


class HandTracker:
    """Utility class for extracting normalized hand landmarks using MediaPipe Tasks API.

    The landmarks are returned as a flat NumPy array of 63 floats (21 points * 3 coordinates).
    Normalization steps:
    1. Translate so the wrist (landmark 0) is at the origin.
    2. Scale by the distance between the wrist and the middle-finger MCP (landmark 9)
       to achieve size invariance.
    """

    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5):
        base_options = mp_python.BaseOptions(model_asset_path=_TASK_FILE)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE if static_image_mode else mp_vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self._running_mode = options.running_mode
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def extract_landmarks(self, image):
        """Extract normalized 63-dimensional landmark vector from a BGR image.

        Parameters
        ----------
        image : np.ndarray
            BGR image as returned by OpenCV.

        Returns
        -------
        tuple (np.ndarray, list) or (None, None)
            First element is flattened (63,) array of normalized landmarks.
            Second element is the raw list of NormalizedLandmark objects.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        if self._running_mode == mp_vision.RunningMode.VIDEO:
            self._timestamp_ms += 33  # ~30 FPS
            result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None, None

        hand_lms = result.hand_landmarks[0]  # First detected hand

        # Collect into NumPy array (21, 3)
        points = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms])

        # Translate wrist (index 0) to origin
        wrist = points[0].copy()
        points -= wrist

        # Scale by wrist-to-middle-finger-MCP distance (landmark 9)
        scale = np.linalg.norm(points[9])
        if scale > 0:
            points /= scale

        return points.flatten(), hand_lms

    def close(self):
        self._landmarker.close()


def _draw_hand_landmarks(image, hand_lms):
    """Draw hand skeleton on image using raw landmark list from Tasks API."""
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    h, w = image.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

    for start, end in CONNECTIONS:
        cv2.line(image, pts[start], pts[end], (0, 220, 255), 2)
    for pt in pts:
        cv2.circle(image, pt, 4, (255, 255, 255), -1)
        cv2.circle(image, pt, 4, (0, 180, 220), 1)


# Simple test when run directly.
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        landmarks, hand_lms = tracker.extract_landmarks(frame)
        display = frame.copy()
        if landmarks is not None:
            cv2.putText(display, "Hand detected", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            _draw_hand_landmarks(display, hand_lms)
        else:
            cv2.putText(display, "No hand", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('HandTracker Demo', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
