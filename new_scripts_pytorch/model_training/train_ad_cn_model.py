"""
Train an AD vs CN 3-D CNN using PyTorch.
This is a *conceptually* faithful rewrite of ``new_scripts/model_training/train_ad_cn_model.py``
minus automatic hyper-parameter tuning – simplicity is prioritised per the
user request.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Add project root to path so that "new_scripts_pytorch" can be imported when the
# script is executed directly via ``python ...``.
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from new_scripts_pytorch.utils.data_utils import load_pickle, create_data_loader
from new_scripts_pytorch.utils.model_utils import (
    create_3d_cnn_model, train_model, evaluate_model
)
from new_scripts_pytorch.utils.visualization_utils import (
    plot_training_history, plot_confusion_matrix, plot_roc_curve
)
from new_scripts_pytorch.utils.config import (
    BATCH_SIZE, DEVICE, SAVED_MODELS_DIR, TENSORBOARD_DIR, RESULTS_DIR, NUM_CLASSES
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_dataloaders(data_pickle_path: str | Path, batch_size: int = BATCH_SIZE):
    """Return train & validation torch ``DataLoader`` objects from *pickle* file."""
    data = load_pickle(data_pickle_path)
    train_loader = create_data_loader(data['train_images'], data['train_labels'], batch_size=batch_size, shuffle=True)
    val_loader = create_data_loader(data['val_images'], data['val_labels'], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def main():
    logger.info("[AD/CN] Starting training pipeline (PyTorch)…")

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    data_pickle = PROJECT_ROOT / 'ad_cn_preprocessed.pkl'
    if not data_pickle.exists():
        logger.error("Expected pre-processed data at %s – aborting", data_pickle)
        sys.exit(1)

    train_loader, val_loader = build_dataloaders(data_pickle)
    logger.info("Loaded %d training & %d validation samples", len(train_loader.dataset), len(val_loader.dataset))

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------
    model = create_3d_cnn_model()
    logger.info("Model instantiated – %.1fK parameters", sum(p.numel() for p in model.parameters()) / 1e3)

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    history = train_model(model, train_loader, val_loader, model_name='ad_cn')

    # Plot training curves
    plot_training_history(history, TENSORBOARD_DIR / 'ad_cn_training_history.png')

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------
    # Build a combined loader for evaluation purposes (no shuffling)
    full_val_loader = create_data_loader(
        load_pickle(data_pickle)['val_images'],
        load_pickle(data_pickle)['val_labels'],
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    metrics = evaluate_model(model, full_val_loader)
    logger.info("Validation metrics → loss: %.4f | acc: %.3f", metrics['test_loss'], metrics['test_accuracy'])

    # Confusion matrix / ROC
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in full_val_loader:
            batch_x = batch_x.to(DEVICE)
            logits = model(batch_x.unsqueeze(1))
            probs = torch.softmax(logits, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(batch_y.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_pred_classes = y_pred.argmax(axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    plot_confusion_matrix(cm, ['CN', 'AD'], RESULTS_DIR / 'ad_cn_confusion_matrix.png')

    fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, RESULTS_DIR / 'ad_cn_roc_curve.png')

    logger.info("[AD/CN] Training pipeline finished successfully.")


if __name__ == "__main__":
    main() 