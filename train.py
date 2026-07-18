import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils.model import FeedForwardNN

# Define PyTorch Dataset
class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_and_evaluate():
    print("==================================================")
    print("        Sign Language Model Training & Eval       ")
    print("==================================================")
    
    csv_path = 'landmarks_dataset.csv'
    if not os.path.exists(csv_path):
        print(f"Error: Dataset file '{csv_path}' not found.")
        print("Please collect data first by running 'data_collection.py'.")
        return
        
    # 1. Load and Parse Data
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} samples.")
    
    # Check classes
    classes = sorted(df['label'].unique())
    num_classes = len(classes)
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    print(f"Classes detected: {classes}")
    
    # 2. Session-Based Split: train / val / test (Preventing Data Leakage)
    # Whole sessions are held out, and the two evaluation roles are separated:
    # val picks the checkpoint and the winning model, test is touched exactly
    # once at the end. Selecting on the same set you report on inflates the
    # headline number.
    train_rows = []
    val_rows = []
    test_rows = []

    # Ensure reproducibility
    np.random.seed(42)

    print(f"\n{'label':<8}{'train':>7}{'val':>6}{'test':>6}  (sessions)")
    for label in classes:
        label_df = df[df['label'] == label]
        sessions = list(label_df['session_id'].unique())
        if len(sessions) < 3:
            raise ValueError(
                f"Label '{label}' has only {len(sessions)} session(s); the "
                f"train/val/test split holds out whole sessions and needs at "
                f"least 3. Collect more with data_collection.py.")
        np.random.shuffle(sessions)

        n = len(sessions)
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        n_train = n - n_val - n_test

        train_sessions = sessions[:n_train]
        val_sessions = sessions[n_train:n_train + n_val]
        test_sessions = sessions[n_train + n_val:]
        print(f"{label:<8}{n_train:>7}{n_val:>6}{n_test:>6}")

        train_rows.append(label_df[label_df['session_id'].isin(train_sessions)])
        val_rows.append(label_df[label_df['session_id'].isin(val_sessions)])
        test_rows.append(label_df[label_df['session_id'].isin(test_sessions)])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    val_df = pd.concat(val_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)

    print(f"\nTrain set: {len(train_df)} samples (from {train_df['session_id'].nunique()} sessions)")
    print(f"Val set:   {len(val_df)} samples (from {val_df['session_id'].nunique()} sessions)")
    print(f"Test set:  {len(test_df)} samples (from {test_df['session_id'].nunique()} sessions)")

    # Extract features (X) and labels (y)
    feature_cols = [c for c in df.columns if c.startswith('l') and c != 'label' and c != 'session_id']
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].map(label_to_idx).values
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].map(label_to_idx).values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].map(label_to_idx).values
    
    # Save the label mapping for deployment
    joblib.dump(classes, 'label_map.joblib')
    
    # 3. Model 1: Baseline Random Forest Classifier
    print("\n--- Training Random Forest (Baseline) ---")
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    rf_val_acc = accuracy_score(y_val, rf_model.predict(X_val))
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest Val Accuracy:  {rf_val_acc:.4f}")
    print(f"Random Forest Test Accuracy: {rf_acc:.4f}")
    
    # Save Random Forest Model
    joblib.dump(rf_model, 'random_forest.joblib')
    
    # 4. Model 2: PyTorch Neural Network
    print("\n--- Training PyTorch Feed-Forward Neural Network ---")
    train_dataset = LandmarkDataset(X_train, y_train)
    val_dataset = LandmarkDataset(X_val, y_val)
    test_dataset = LandmarkDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    nn_model = FeedForwardNN(input_dim=len(feature_cols), output_dim=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
    
    # Training Loop — checkpoint selection uses the VAL set only, so the test
    # set stays untouched until the single final evaluation below.
    epochs = 30
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        nn_model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = nn_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)

        # Evaluate on Val Set for checkpoint selection
        nn_model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = nn_model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == targets).item()

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = correct / len(val_dataset)

        # Save model weights if it improves val loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(nn_model.state_dict(), 'best_model.pth')

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
            
    # Load best (val-selected) weights and evaluate once on val and test
    nn_model.load_state_dict(torch.load('best_model.pth'))
    nn_model.eval()

    def _predict(loader):
        preds = []
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                outputs = nn_model(inputs)
                _, p = torch.max(outputs, 1)
                preds.extend(p.cpu().numpy())
        return preds

    nn_val_acc = accuracy_score(y_val, _predict(val_loader))
    nn_preds = _predict(test_loader)
    nn_acc = accuracy_score(y_test, nn_preds)
    print(f"\nPyTorch NN Val Accuracy:  {nn_val_acc:.4f}")
    print(f"PyTorch NN Test Accuracy: {nn_acc:.4f}")

    # 5. Model Evaluation & Comparison
    print("\n==================================================")
    print("                Model Evaluation                  ")
    print("==================================================")
    print(f"{'':<16}{'Val':>8}{'Test':>8}")
    print(f"{'Random Forest':<16}{rf_val_acc:>8.4f}{rf_acc:>8.4f}")
    print(f"{'Neural Network':<16}{nn_val_acc:>8.4f}{nn_acc:>8.4f}")

    # Select the winning model on VAL accuracy; its TEST accuracy is the
    # honest headline number (picking the winner on test would reuse the
    # test set for selection).
    winner_name = "Random Forest" if rf_val_acc >= nn_val_acc else "Neural Network"
    winner_preds = rf_preds if rf_val_acc >= nn_val_acc else nn_preds
    print(f"\nWinning Model (by val): {winner_name}")
    
    # Print detailed classification report for the winning model
    y_test_labels = [idx_to_label[i] for i in y_test]
    winner_preds_labels = [idx_to_label[i] for i in winner_preds]
    
    print(f"\nClassification Report for {winner_name}:")
    print(classification_report(y_test_labels, winner_preds_labels, target_names=classes))
    
    # Compute and plot Confusion Matrix for the winning model
    cm = confusion_matrix(y_test_labels, winner_preds_labels, labels=classes)
    # Scale the figure with the class count so 20+ classes stay readable
    side = max(8, 0.5 * num_classes + 4)
    plt.figure(figsize=(side, side * 0.8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {winner_name} (Session Held-Out)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save Confusion Matrix Image
    cm_path = 'evaluation_confusion_matrix.png'
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix heatmap saved to: {os.path.abspath(cm_path)}")
    print("==================================================")

if __name__ == "__main__":
    train_and_evaluate()
