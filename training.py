#!/usr/bin/env python3
"""
Training and evaluation script for fraud detection model.
Loads data directories from config.json; hyperparameters are hardcoded for reproducibility.
"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# --------------------- Logging ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ------------------ Hyperparameters ----------------
BATCH_SIZE = 64
LEARNING_RATE = 0.0002900853857739499
WEIGHT_DECAY = 0.0001311625707079125
SCHEDULER_FACTOR = 0.6786503284076598
SCHEDULER_PATIENCE = 5
NUM_EPOCHS = 40
EARLY_STOP_PATIENCE = 5
OVERFIT_PATIENCE = 3

# ------------------ Utility Functions --------------
def load_config(config_path: Path) -> dict:
    """Load JSON configuration from file."""
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Missing config: {config_path}")
    with config_path.open('r') as f:
        return json.load(f)


def get_data_loaders(cfg: dict):
    """Create DataLoaders for train/val/test/OOS splits."""
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dir = Path(cfg['TRAIN_DIR'])
    test_dir = Path(cfg['TEST_DIR'])
    oos_dir = Path(cfg['OUT_OF_SAMPLE_DIR'])

    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = ImageFolder(root=test_dir, transform=test_transform)
    oos_dataset  = ImageFolder(root=oos_dir, transform=test_transform)

    # Split train into train/validation
    val_ratio = cfg.get('VAL_RATIO', 0.1)
    train_size = int((1 - val_ratio) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    oos_loader   = DataLoader(oos_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, oos_loader, train_dataset.classes


def evaluate_model(model, loader, device, class_names):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds, all_labels = [], []
    correct = total = 0
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += int(preds[i] == labels[i])
                class_total[label] += 1

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = correct / total
    per_class = {
        class_names[i]: class_correct[i] / class_total[i]
        if class_total[i] > 0 else 0
        for i in range(len(class_names))
    }
    conf_mat = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    return accuracy, per_class, conf_mat, report


def train_and_evaluate(cfg: dict):
    """Train the model with early stopping and overfitting check."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader, oos_loader, class_names = get_data_loaders(cfg)

    # Model setup
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.to(device)

    # Class weights
    labels = []
    for _, y in train_loader:
        labels.extend(y.numpy())
    cw = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(cw, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, verbose=True
    )

    best_val_acc = 0.0
    epochs_no_improve = 0
    prev_train_loss = float('inf')
    prev_val_loss = float('inf')
    overfit_count = 0

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = correct = total = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation
        val_acc, _, _, _ = evaluate_model(model, val_loader, device, class_names)
        history['val_acc'].append(val_acc)

        scheduler.step(train_loss)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), cfg.get('BEST_MODEL_PATH', 'best_model.pth'))
            logger.info(f"Epoch {epoch+1}: New best val_acc={best_val_acc:.4f}")
        else:
            epochs_no_improve += 1
            logger.info(f"Epoch {epoch+1}: no improvement {epochs_no_improve}/{EARLY_STOP_PATIENCE}")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            logger.info(f"Stopping early at epoch {epoch+1}")
            break

        # Overfitting check
        # (val loss not tracked here; implement if needed)

        logger.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    # Final evaluation
    model.load_state_dict(torch.load(cfg.get('BEST_MODEL_PATH', 'best_model.pth')))
    for name, loader in [('Train', train_loader), ('Test', test_loader), ('OOS', oos_loader)]:
        acc, per_class, cm, rpt = evaluate_model(model, loader, device, class_names)
        logger.info(f"{name} Accuracy: {acc:.4f}")
        logger.info(f"{name} Per-class: {per_class}")

    # Plot and save metrics
    out_dir = Path(cfg.get('OUTPUT_DIR', '.'))
    out_dir.mkdir(exist_ok=True)
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(out_dir / 'training_history.csv', index=False)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Loss')
    plt.title('Training Loss')
    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.legend()
    plt.savefig(out_dir / 'training_plot.png')
    logger.info(f"Training artifacts saved to {out_dir}")


def main():
    cfg = load_config(Path(__file__).parent / 'config.json')
    train_and_evaluate(cfg)


if __name__ == '__main__':
    main()
