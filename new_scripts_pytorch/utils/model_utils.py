"""
PyTorch equivalents of ``new_scripts.utils.model_utils``.
Only the subset of functionality necessary for the ported training / evaluation
pipeline is implemented – namely model creation, training, evaluation and
(weight) persistence helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from .config import (
    SAVED_MODELS_DIR, LEARNING_RATE, EPOCHS, NUM_CLASSES, EARLY_STOPPING_PATIENCE, DEVICE,
)

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------

class Simple3DCNN(nn.Module):
    """A *very* small 3-D CNN roughly mirroring the Keras counterpart."""

    def __init__(self, input_channels: int = 1, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 12 * 12, 512),  # assumes input size (96,96,96)
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_3d_cnn_model(input_channels: int = 1, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Factory wrapper keeping API parity with the original code base."""
    model = Simple3DCNN(input_channels=input_channels, num_classes=num_classes)
    return model.to(DEVICE)

# -----------------------------------------------------------------------------
# Training & evaluation helpers
# -----------------------------------------------------------------------------

def _run_epoch(model: nn.Module, loader: DataLoader, criterion, optimiser=None) -> Tuple[float, float]:
    """Run one training or validation epoch and return (loss, accuracy)."""
    is_train = optimiser is not None
    model.train(is_train)
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        if is_train:
            optimiser.zero_grad(set_to_none=True)
        logits = model(batch_x.unsqueeze(1))  # add channel dim
        loss = criterion(logits, batch_y)
        if is_train:
            loss.backward()
            optimiser.step()
        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == batch_y).sum().item()
        total_samples += batch_x.size(0)

    return total_loss / total_samples, total_correct / total_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
) -> Dict[str, List[float]]:
    """Train *model* using a simple early-stopping loop."""
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }

    best_val_acc = 0.0
    patience_counter = 0
    save_path = SAVED_MODELS_DIR / f"{model_name}.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimiser)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimiser=None)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        logger.info(
            "Epoch %d/%d — train_loss: %.4f | train_acc: %.3f | val_loss: %.4f | val_acc: %.3f",
            epoch, EPOCHS, train_loss, train_acc, val_loss, val_acc,
        )

        # Early stopping logic --------------------------------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered (no val improvement for %d epochs)", EARLY_STOPPING_PATIENCE)
                break

    # Load best weights before returning
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return history


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
    """Return test loss/accuracy metrics for *model*."""
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = _run_epoch(model, test_loader, criterion, optimiser=None)
    return {'test_loss': test_loss, 'test_accuracy': test_acc}


# -----------------------------------------------------------------------------
# Misc helpers
# -----------------------------------------------------------------------------

def load_trained_model(model_path: Union[str, Path], input_channels: int = 1, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Instantiate a model and load pre-trained weights from *model_path*."""
    model = create_3d_cnn_model(input_channels, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model 

# -----------------------------------------------------------------------------
# Monte-Carlo dropout helpers (uncertainty estimation)
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns


def monte_carlo_predictions(
    model: nn.Module,
    data: np.ndarray,
    num_samples: int = 100,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return *(mean_pred, std_pred)* over *num_samples* stochastic forward passes.

    ``data`` is a numpy array shaped (N, D, H, W) (depth/height/width). We add the
    required channel dimension on-the-fly.
    """
    model = model.to(DEVICE)
    model.eval()  # ensure eval before toggling to train inside loop

    # Create a DataLoader once to avoid repeated CPU→GPU transfers of the whole
    # dataset. We keep shuffle=False so sample order is stable across passes.
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch.from_numpy(data).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds: List[np.ndarray] = []

    with torch.no_grad():
        for _ in range(num_samples):
            model.train()  # enable dropout
            preds_batch: List[np.ndarray] = []
            for (batch_x,) in loader:
                batch_x = batch_x.to(DEVICE)
                logits = model(batch_x.unsqueeze(1))
                probs = torch.softmax(logits, dim=1)
                preds_batch.append(probs[:, 1].cpu().numpy())  # probability of class 1
            all_preds.append(np.concatenate(preds_batch))

    preds = np.stack(all_preds, axis=0)  # (num_samples, N)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)
    return mean_pred, std_pred


def analyze_prediction_uncertainty(
    mean_pred: np.ndarray,
    std_pred: np.ndarray,
    true_labels: np.ndarray,
    save_path: Union[str, Path],
) -> None:
    """Replicates original Keras helper – produces error-bar plot and logs stats."""
    try:
        plt.figure(figsize=(12, 6))

        sort_idx = np.argsort(true_labels)
        mean_pred_sorted = mean_pred[sort_idx]
        std_pred_sorted = std_pred[sort_idx]
        true_sorted = true_labels[sort_idx]

        plt.errorbar(
            range(len(true_sorted)), mean_pred_sorted,
            yerr=std_pred_sorted, fmt='o', alpha=0.5,
            label='Predictions ± Uncertainty',
        )
        plt.plot(true_sorted, 'r-', label='True Labels')
        plt.title('Monte Carlo Predictions with Uncertainty')
        plt.xlabel('Sample Index')
        plt.ylabel('Prediction (probability)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        # Log simple statistics
        logger.info(
            "Uncertainty → mean: %.4f | std: %.4f | max: %.4f | min: %.4f",
            std_pred.mean(), std_pred.std(), std_pred.max(), std_pred.min(),
        )
    except Exception as e:
        logger.error("Error plotting prediction uncertainty: %s", e)
        raise 