import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features.audio_processing import TorgoDataset
from src.models.baseline import DysarthriaCRNN
from src.training.train import get_dataloaders
def plot_training_history(history_path="results/training_history.csv"):
    if not os.path.exists(history_path):
        print("History CSV not found. Skipping history plots.")
        return
    df = pd.read_csv(history_path)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
    plt.plot(df['epoch'], df['val_acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/training_curves.png")
    print("Saved training curves to results/training_curves.png")
def evaluate_model():
    metadata_path = "data/processed/torgo_metadata.csv"
    if not os.path.exists(metadata_path):
        print("Metadata not found!")
        return
    os.makedirs("results", exist_ok=True)
    plot_training_history()
    _, val_loader = get_dataloaders(metadata_path, batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = DysarthriaCRNN(num_classes=4).to(device)
    
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        print("Model weights not found! Train the model first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print("Evaluating...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(1)
                
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    target_names = ['Normal', 'Mild', 'Moderate', 'Severe']
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=target_names))
    
    # Confusion Matrix (Raw)
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', 
                xticklabels=target_names, yticklabels=target_names,
                annot_kws={"size": 14})
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix (Total Counts)', fontsize=14)
    plt.savefig("results/confusion_matrix.png")
    print("Saved raw confusion matrix to results/confusion_matrix.png")
    
    # Confusion Matrix (Normalized)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='viridis', 
                xticklabels=target_names, yticklabels=target_names,
                annot_kws={"size": 14})
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Normalized Confusion Matrix', fontsize=14)
    plt.savefig("results/confusion_matrix_norm.png")
    print("Saved normalized confusion matrix to results/confusion_matrix_norm.png")
if __name__ == "__main__":
    evaluate_model()
