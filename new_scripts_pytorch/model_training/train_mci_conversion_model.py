"""
Train MCI conversion (pMCI vs sMCI) classifier with PyTorch and 5-fold
cross-validation. This is a straightforward rewrite of
``new_scripts/model_training/train_mci_conversion_model.py`` but without hyper-
parameter tuning – we reuse the same 3-D CNN architecture from
``utils.model_utils`` and focus on simplicity.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from new_scripts_pytorch.utils.data_utils import (
    load_pickle,
    create_data_loader,
)
from new_scripts_pytorch.utils.model_utils import (
    create_3d_cnn_model,
    train_model,
    evaluate_model,
)
from new_scripts_pytorch.utils.visualization_utils import (
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
)
from new_scripts_pytorch.utils.config import (
    DEVICE,
    BATCH_SIZE,
    SAVED_MODELS_DIR,
    TENSORBOARD_DIR,
    RESULTS_DIR,
    NUM_CLASSES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_kfold_data(pkl_path: Path):
    """Return list of dicts with numpy arrays from ``mci_kfold_datasets.pkl``."""
    folds: List[Dict[str, np.ndarray]] = load_pickle(pkl_path)
    return folds


def main():
    logger.info("[MCI Conversion] Starting k-fold training pipeline (PyTorch)…")

    data_pickle = PROJECT_ROOT / 'mci_kfold_datasets.pkl'
    if not data_pickle.exists():
        logger.error("%s not found – please run create_kfold_datasets first", data_pickle)
        sys.exit(1)

    folds = load_kfold_data(data_pickle)
    all_fold_metrics = []

    for i, fold_dict in enumerate(folds, start=1):
        logger.info("--- Fold %d/%d ---", i, len(folds))

        train_loader = create_data_loader(
            fold_dict['train_images'], fold_dict['train_labels'], batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = create_data_loader(
            fold_dict['val_images'], fold_dict['val_labels'], batch_size=BATCH_SIZE, shuffle=False
        )

        model = create_3d_cnn_model()

        history = train_model(model, train_loader, val_loader, model_name=f'mci_conv_fold_{i}')
        plot_training_history(history, TENSORBOARD_DIR / f'mci_conv_fold_{i}_training_history.png')

        # Evaluation ------------------------------------------------------
        metrics = evaluate_model(model, val_loader)
        logger.info("Fold %d metrics → loss: %.4f | acc: %.3f", i, metrics['test_loss'], metrics['test_accuracy'])
        all_fold_metrics.append(metrics)

        # Confusion matrix & ROC -----------------------------------------
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                probs = torch.softmax(model(batch_x.to(DEVICE).unsqueeze(1)), dim=1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch_y.numpy())
        y_pred = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels)
        y_pred_classes = y_pred.argmax(axis=1)

        cm = confusion_matrix(y_true, y_pred_classes)
        plot_confusion_matrix(cm, ['sMCI', 'pMCI'], RESULTS_DIR / f'mci_conv_fold_{i}_cm.png')

        fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
        roc_auc = auc(fpr, tpr)
        plot_roc_curve(fpr, tpr, roc_auc, RESULTS_DIR / f'mci_conv_fold_{i}_roc.png')

    # Aggregate metrics -------------------------------------------------------
    import pandas as pd

    metrics_df = pd.DataFrame(all_fold_metrics)
    metrics_df.to_csv(RESULTS_DIR / 'mci_conversion_kfold_metrics.csv', index=False)
    logger.info("K-fold training complete. Mean acc: %.3f ± %.3f", metrics_df['test_accuracy'].mean(), metrics_df['test_accuracy'].std())


if __name__ == "__main__":
    main() 