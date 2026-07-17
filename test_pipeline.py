import unittest
import numpy as np
import os
import pandas as pd
import torch
import joblib

# Import components
from utils.hand_tracker import HandTracker
from train import FeedForwardNN, LandmarkDataset

class TestSignLanguagePipeline(unittest.TestCase):
    
    def test_hand_tracker_normalization(self):
        """Test that HandTracker translates the wrist to origin and scales the hand size to 1.0."""
        # Initialize tracker
        tracker = HandTracker()
        
        # Mock hand: 21 points of size (21, 3)
        # Landmark 0 (wrist) at (10.0, 20.0, 30.0)
        # Landmark 9 (middle MCP) at (10.0, 25.0, 30.0) -> Distance from wrist is 5.0
        mock_landmarks = []
        for i in range(21):
            if i == 0:
                mock_landmarks.append((10.0, 20.0, 30.0))
            elif i == 9:
                mock_landmarks.append((10.0, 25.0, 30.0))
            else:
                mock_landmarks.append((10.0 + i, 20.0 + i, 30.0 + i))
                
        # MediaPipe Tasks API mock representation. The HandLandmarker returns a
        # result whose `.hand_landmarks` is a list (one entry per detected hand),
        # each entry being a flat list of landmark objects exposing x/y/z.
        class MockLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        class MockResults:
            def __init__(self, points):
                self.hand_landmarks = [[MockLandmark(p[0], p[1], p[2]) for p in points]]

        # Override the Tasks-API detector to return our mock hand. A default
        # HandTracker runs in VIDEO mode, so detect_for_video is the call used.
        tracker._landmarker.detect_for_video = lambda mp_image, timestamp_ms: MockResults(mock_landmarks)
        
        # Run extraction on a blank image
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        flat_landmarks, hand_lms = tracker.extract_landmarks(blank_img)
        
        # Reshape to (21, 3) for easier asserting
        coords = flat_landmarks.reshape(21, 3)
        
        # 1. Assert wrist translated to origin (0, 0, 0)
        np.testing.assert_array_almost_equal(coords[0], [0.0, 0.0, 0.0])
        
        # 2. Assert wrist-to-middle-MCP distance is scaled to exactly 1.0
        # Landmark 9 coordinate is index 9
        mcp_coords = coords[9]
        distance = np.linalg.norm(mcp_coords)
        self.assertAlmostEqual(distance, 1.0, places=5)
        
    def test_pytorch_dataset(self):
        """Test LandmarkDataset loads features and targets correctly."""
        X = np.random.rand(10, 63)
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        
        dataset = LandmarkDataset(X, y)
        self.assertEqual(len(dataset), 10)
        
        features, target = dataset[0]
        self.assertEqual(features.shape, (63,))
        self.assertEqual(target.item(), 0)
        
    def test_neural_network_forward(self):
        """Test PyTorch Neural Network forward pass outputs correct shape."""
        model = FeedForwardNN(input_dim=63, output_dim=5)
        mock_input = torch.randn(4, 63)
        output = model(mock_input)
        self.assertEqual(output.shape, (4, 5))

    def test_train_pipeline_dry_run(self):
        """Creates a tiny dummy CSV dataset and runs train.py to verify splitting and fitting.

        The whole dry run executes inside a throwaway temporary directory, so it can
        never touch, overwrite, or delete the repository's real dataset or its
        committed model artifacts. Cleanup only removes the temp directory itself.
        """
        import tempfile
        import shutil

        original_cwd = os.getcwd()
        tmp_dir = tempfile.mkdtemp(prefix="train_dryrun_")
        try:
            # train.py reads/writes all paths relative to the cwd, so isolate it here.
            os.chdir(tmp_dir)

            # Create a tiny mock dataset
            # 5 classes (A-E), 8 sessions per class, 15 frames per session
            data = []
            classes = ['A', 'B', 'C', 'D', 'E']
            for label in classes:
                for s in range(1, 9):
                    session_id = f"{label}_{s:02d}"
                    for f in range(15):
                        # 63 landmarks + label + session_id
                        row = list(np.random.rand(63)) + [label, session_id]
                        data.append(row)

            # Write to CSV (inside the temp dir)
            cols = []
            for i in range(21):
                cols.extend([f"l{i}_x", f"l{i}_y", f"l{i}_z"])
            cols.extend(["label", "session_id"])

            df = pd.DataFrame(data, columns=cols)
            df.to_csv('landmarks_dataset.csv', index=False)

            # Run train.py logic directly; every output lands in the temp dir.
            from train import train_and_evaluate
            train_and_evaluate()

            # Verify models generated
            self.assertTrue(os.path.exists('random_forest.joblib'))
            self.assertTrue(os.path.exists('best_model.pth'))
            self.assertTrue(os.path.exists('label_map.joblib'))
            self.assertTrue(os.path.exists('evaluation_confusion_matrix.png'))

        finally:
            # Only the temp directory is removed; real repo files are never referenced.
            os.chdir(original_cwd)
            shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
