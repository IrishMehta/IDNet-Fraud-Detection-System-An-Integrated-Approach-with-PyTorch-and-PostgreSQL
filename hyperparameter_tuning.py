#!/usr/bin/env python3
"""
Hyperparameter tuning for image classification using Optuna.

"""
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from sklearn.utils.class_weight import compute_class_weight
import optuna

# Logging setup
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

logger = configure_logging()


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Missing config: {config_path}")
    with config_path.open('r') as f:
        cfg= json.load(f)
        return cfg["hpo"]


def get_data_loaders(cfg: dict):
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(0.2, 0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_folder = Path(cfg['TRAIN_DIR'])
    dataset = ImageFolder(root=train_folder, transform=train_transform)

    # Split train into train/val
    val_ratio = cfg.get('VAL_RATIO', 0.1)
    train_size = int((1 - val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    return train_ds, val_ds


def objective(trial, train_ds, val_ds, device, num_classes, num_epochs):
    # Hard-coded hyperparameter search space
    batch_size        = trial.suggest_categorical('batch_size', [16, 32, 64])
    lr                = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay      = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    scheduler_factor  = trial.suggest_float('scheduler_factor', 0.1, 0.7)
    scheduler_patience= trial.suggest_int('scheduler_patience', 2, 7)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    # Class weights
    labels = []
    for _, y in train_loader:
        labels.extend(y.cpu().numpy())
    cw = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(cw, dtype=torch.float, device=device)

    # Criterion, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor,
        patience=scheduler_patience, verbose=True
    )

    best_acc = 0.0
    for epoch in range(num_epochs):
        # Training
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total

        # Scheduler step
        scheduler.step(1 - val_acc)

        # Pruning
        trial.report(1 - val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Track best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    return 1 - best_acc


def run_tuning(cfg: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds, val_ds = get_data_loaders(cfg)
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    study.optimize(
        lambda t: objective(
            t, train_ds, val_ds, device,
            cfg.get('NUM_CLASSES', 2), cfg.get('NUM_EPOCHS', 30)
        ),
        n_trials=cfg.get('N_TRIALS', 20)
    )
    logger.info(f"Best params: {study.best_trial.params}")

    # Save best parameters to JSON
    out_dir = Path(cfg.get('OUTPUT_DIR', '.'))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'best_hyperparams.json', 'w') as f:
        json.dump(study.best_trial.params, f, indent=2)
    logger.info(f"Saved best hyperparameters to {out_dir / 'best_hyperparams.json'}")


def main():
    cfg = load_config(Path(__file__).parent / 'config.json')
    run_tuning(cfg)


if __name__ == '__main__':
    main()
    