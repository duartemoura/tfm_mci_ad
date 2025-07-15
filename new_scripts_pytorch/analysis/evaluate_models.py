"""
Evaluate trained PyTorch models and generate basic performance plots. This is a
minimal rewrite of the TensorFlow-based ``new_scripts/analysis/evaluate_models.py``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve,
    average_precision_score,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from new_scripts_pytorch.utils.data_utils import load_pickle, create_data_loader
from new_scripts_pytorch.utils.model_utils import load_trained_model
from new_scripts_pytorch.utils.visualization_utils import (
    plot_confusion_matrix, plot_roc_curve
)
from new_scripts_pytorch.utils.config import RESULTS_DIR, DEVICE, NUM_CLASSES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model_path: Path, data_pickle: Path, model_name: str):
    logger.info("Evaluating model %s", model_name)

    model = load_trained_model(model_path)

    data = load_pickle(data_pickle)
    test_loader = create_data_loader(data['test_images'], data['test_labels'], batch_size=32, shuffle=False)

    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            probs = torch.softmax(model(x.to(DEVICE).unsqueeze(1)), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())

    y_pred = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    y_pred_classes = y_pred.argmax(axis=1)

    # Metrics ----------------------------------------------------------------
    report = classification_report(y_true, y_pred_classes, output_dict=True)
    fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_pred[:, 1])
    avg_precision = average_precision_score(y_true, y_pred[:, 1])

    metrics: Dict[str, float] = {
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        **{k: v for k, v in report.items() if isinstance(v, float)},
    }

    # Save metrics to CSV
    out_dir = RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame([metrics]).to_csv(out_dir / 'evaluation_metrics.csv', index=False)

    # Plots
    cm = confusion_matrix(y_true, y_pred_classes)
    plot_confusion_matrix(cm, [f'Class {i}' for i in range(NUM_CLASSES)], out_dir / 'confusion_matrix.png')
    plot_roc_curve(fpr, tpr, roc_auc, out_dir / 'roc_curve.png')

    logger.info("Evaluation finished – AUC: %.3f | AP: %.3f", roc_auc, avg_precision)


def main():
    # Example: AD/CN model located at saved_models_pytorch/ad_cn.pt
    evaluate_model(PROJECT_ROOT / 'saved_models_pytorch' / 'ad_cn.pt', PROJECT_ROOT / 'ad_cn_test.pkl', 'ad_cn_model')


if __name__ == "__main__":
    main() 