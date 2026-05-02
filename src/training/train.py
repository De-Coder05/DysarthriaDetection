import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features.audio_processing import TorgoDataset
from src.models.baseline import DysarthriaCRNN
def get_dataloaders(metadata_path, batch_size=32):
    df = pd.read_csv(metadata_path)

    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['severity_label']
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
    print("Validation Class Distribution:")
    print(val_df['severity_label'].value_counts().sort_index())
    train_dataset = TorgoDataset(train_df)
    val_dataset = TorgoDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
def train_model(epochs=20, batch_size=32, lr=0.001, patience=3):
    metadata_path = "data/processed/torgo_metadata.csv"
    if not os.path.exists(metadata_path):
        print("Metadata not found! Run parse_torgo.py first.")
        return
    train_loader, val_loader = get_dataloaders(metadata_path, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(metadata_path)
    class_counts = df['severity_label'].value_counts().sort_index().values
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    weights = weights / weights.sum() * 4.0
    weights = weights.to(device)

    model = DysarthriaCRNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        train_acc = 100. * correct / total
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if len(inputs.shape) == 3:
                    inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        val_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_acc
        })
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/best_model.pth")
            print("Saved new best model.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    pd.DataFrame(history).to_csv("results/training_history.csv", index=False)
    print("Saved training history to results/training_history.csv")
if __name__ == "__main__":
    train_model(epochs=20, patience=3)
