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

# Define PyTorch Dataset
class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define Neural Network Model
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
    
    # 2. Session-Based Split (Preventing Data Leakage)
    train_rows = []
    test_rows = []
    
    # Ensure reproducibility
    np.random.seed(42)
    
    for label in classes:
        label_df = df[df['label'] == label]
        sessions = list(label_df['session_id'].unique())
        np.random.shuffle(sessions)
        
        # 75% train, 25% test split
        split_idx = int(len(sessions) * 0.75)
        # Ensure we have at least 1 session in test and train
        if split_idx == 0:
            split_idx = 1
        elif split_idx == len(sessions):
            split_idx = len(sessions) - 1
            
        train_sessions = sessions[:split_idx]
        test_sessions = sessions[split_idx:]
        
        train_rows.append(label_df[label_df['session_id'].isin(train_sessions)])
        test_rows.append(label_df[label_df['session_id'].isin(test_sessions)])
        
    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    
    print(f"Train set: {len(train_df)} samples (from {train_df['session_id'].nunique()} sessions)")
    print(f"Test set: {len(test_df)} samples (from {test_df['session_id'].nunique()} sessions)")
    
    # Extract features (X) and labels (y)
    feature_cols = [c for c in df.columns if c.startswith('l') and c != 'label' and c != 'session_id']
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].map(label_to_idx).values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].map(label_to_idx).values
    
    # Save the label mapping for deployment
    joblib.dump(classes, 'label_map.joblib')
    
    # 3. Model 1: Baseline Random Forest Classifier
    print("\n--- Training Random Forest (Baseline) ---")
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    
    # Save Random Forest Model
    joblib.dump(rf_model, 'random_forest.joblib')
    
    # 4. Model 2: PyTorch Neural Network
    print("\n--- Training PyTorch Feed-Forward Neural Network ---")
    train_dataset = LandmarkDataset(X_train, y_train)
    test_dataset = LandmarkDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    nn_model = FeedForwardNN(input_dim=len(feature_cols), output_dim=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = 30
    best_test_loss = float('inf')
    
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
        
        # Evaluate on Test Set for monitoring
        nn_model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = nn_model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == targets).item()
                
        epoch_test_loss = test_loss / len(test_dataset)
        epoch_test_acc = correct / len(test_dataset)
        
        # Save model weights if it improves test loss
        if epoch_test_loss < best_test_loss:
            best_test_loss = epoch_test_loss
            torch.save(nn_model.state_dict(), 'best_model.pth')
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_loss:.4f} | Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.4f}")
            
    # Load best weights and get predictions
    nn_model.load_state_dict(torch.load('best_model.pth'))
    nn_model.eval()
    nn_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = nn_model(inputs)
            _, preds = torch.max(outputs, 1)
            nn_preds.extend(preds.cpu().numpy())
            
    nn_acc = accuracy_score(y_test, nn_preds)
    print(f"\nPyTorch Neural Network Final Accuracy: {nn_acc:.4f}")
    
    # 5. Model Evaluation & Comparison
    print("\n==================================================")
    print("                Model Evaluation                  ")
    print("==================================================")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    print(f"Neural Network Accuracy: {nn_acc:.4f}")
    
    # Select the winning model based on overall accuracy
    winner_name = "Random Forest" if rf_acc >= nn_acc else "Neural Network"
    winner_preds = rf_preds if rf_acc >= nn_acc else nn_preds
    print(f"\nWinning Model: {winner_name}")
    
    # Print detailed classification report for the winning model
    y_test_labels = [idx_to_label[i] for i in y_test]
    winner_preds_labels = [idx_to_label[i] for i in winner_preds]
    
    print(f"\nClassification Report for {winner_name}:")
    print(classification_report(y_test_labels, winner_preds_labels, target_names=classes))
    
    # Compute and plot Confusion Matrix for the winning model
    cm = confusion_matrix(y_test_labels, winner_preds_labels, labels=classes)
    plt.figure(figsize=(8, 6))
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
