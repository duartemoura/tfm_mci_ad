"""
Analyze model predictive uncertainty using Monte-Carlo dropout in PyTorch.
This file is a direct rewrite of ``new_scripts/analysis/analyze_uncertainty.py``
with TensorFlow/Keras calls replaced by the corresponding helpers from
``new_scripts_pytorch.utils.model_utils``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from new_scripts_pytorch.utils.data_utils import load_pickle
from new_scripts_pytorch.utils.model_utils import (
    load_trained_model,
    monte_carlo_predictions,
    analyze_prediction_uncertainty,
)
from new_scripts_pytorch.utils.config import RESULTS_DIR, SAVED_MODELS_DIR, DEVICE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_model_uncertainty(
    model_path: Path,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    n_samples: int = 100,
    save_dir: Path | None = None,
) -> Dict[str, Any]:
    """Run MC-dropout on *model* and compute basic uncertainty statistics."""
    save_dir = save_dir or RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    model = load_trained_model(model_path)
    mean_preds, std_preds = monte_carlo_predictions(model, test_data, num_samples=n_samples)

    # ------------------------------------------------------------------
    # Visual summary
    # ------------------------------------------------------------------
    analyze_prediction_uncertainty(mean_preds, std_preds, test_labels, save_dir / 'mc_predictions_uncertainty.png')

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    uncertainty = std_preds
    prediction_errors = np.abs(mean_preds - test_labels)
    metrics: Dict[str, Any] = {
        'mean_uncertainty': float(uncertainty.mean()),
        'std_uncertainty': float(uncertainty.std()),
        'max_uncertainty': float(uncertainty.max()),
        'min_uncertainty': float(uncertainty.min()),
        'uncertainty_error_correlation': float(np.corrcoef(uncertainty, prediction_errors)[0, 1]),
    }

    pd.DataFrame([metrics]).to_csv(save_dir / 'uncertainty_metrics.csv', index=False)

    # Distribution of uncertainty
    plt.figure(figsize=(10, 6))
    sns.histplot(uncertainty, bins=30)
    plt.title('Distribution of Prediction Uncertainty')
    plt.xlabel('Uncertainty (Std Dev)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_dir / 'uncertainty_distribution.png')
    plt.close()

    # Uncertainty vs error
    plt.figure(figsize=(10, 6))
    plt.scatter(uncertainty, prediction_errors, alpha=0.5)
    plt.title('Uncertainty vs Prediction Error')
    plt.xlabel('Uncertainty (Std Dev)')
    plt.ylabel('Absolute Prediction Error')
    plt.tight_layout()
    plt.savefig(save_dir / 'uncertainty_vs_error.png')
    plt.close()

    return metrics


def main():
    logger.info('Starting PyTorch uncertainty analysis…')

    # ------------------------------------------------------------------
    # I/O paths
    # ------------------------------------------------------------------
    results_root = RESULTS_DIR / 'uncertainty_analysis'
    results_root.mkdir(parents=True, exist_ok=True)

    # Load test data (expects same pickle structure as TF version)
    test_data = load_pickle(PROJECT_ROOT / 'test_data.pkl')
    test_labels = load_pickle(PROJECT_ROOT / 'test_labels.pkl')

    # AD/CN model --------------------------------------------------------
    ad_cn_dir = results_root / 'ad_cn'
    ad_cn_dir.mkdir(parents=True, exist_ok=True)
    ad_cn_metrics = analyze_model_uncertainty(
        SAVED_MODELS_DIR / 'ad_cn.pt', test_data, test_labels, save_dir=ad_cn_dir
    )

    # (Optional) other models, e.g. MCI conversion ----------------------
    mci_dir = results_root / 'mci_conversion'
    mci_dir.mkdir(parents=True, exist_ok=True)
    try:
        mci_metrics = analyze_model_uncertainty(
            SAVED_MODELS_DIR / 'mci_conversion.pt', test_data, test_labels, save_dir=mci_dir
        )
    except FileNotFoundError:
        logger.warning('MCI conversion model not found – skipping')
        mci_metrics = {}

    # Compare metrics ----------------------------------------------------
    comparison_df = pd.DataFrame([
        {'Model': 'AD/CN', **ad_cn_metrics},
        {'Model': 'MCI Conversion', **mci_metrics} if mci_metrics else {},
    ])
    comparison_df.to_csv(results_root / 'model_uncertainty_comparison.csv', index=False)

    logger.info('Uncertainty analysis complete.')


if __name__ == '__main__':
    main() 