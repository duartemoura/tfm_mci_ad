"""
Analyze model uncertainty using Monte Carlo dropout with PyTorch.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from new_scripts_pytorch.utils.data_utils import load_pickle, create_data_loader
from new_scripts_pytorch.utils.model_utils import create_3d_cnn_model, load_trained_model, monte_carlo_predictions
from new_scripts_pytorch.utils.config import RESULTS_DIR, SAVED_MODELS_DIR, BATCH_SIZE, PICKLE_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_model_uncertainty(model: nn.Module,
                            test_loader: DataLoader,
                            test_labels: np.ndarray,
                            n_samples: int = 100,
                            save_dir: Path = None) -> Dict[str, Any]:
    """Analyze model uncertainty using Monte Carlo dropout."""
    try:
        mean_preds, std_preds = monte_carlo_predictions(model, test_loader, num_samples=n_samples)
        
        uncertainty = std_preds.mean(axis=1) # Average uncertainty across classes
        uncertainty_metrics = {
            'mean_uncertainty': np.mean(uncertainty),
            'std_uncertainty': np.std(uncertainty),
            'max_uncertainty': np.max(uncertainty),
            'min_uncertainty': np.min(uncertainty)
        }
        
        # Taking argmax of mean predictions for error calculation
        pred_labels = np.argmax(mean_preds, axis=1)
        prediction_errors = np.abs(pred_labels - test_labels)
        uncertainty_correlation = np.corrcoef(uncertainty, prediction_errors)[0, 1]
        uncertainty_metrics['uncertainty_error_correlation'] = uncertainty_correlation

        # Plots and saves
        plt.figure(figsize=(10, 6))
        sns.histplot(uncertainty, bins=30, kde=True)
        plt.title('Distribution of Prediction Uncertainty')
        plt.xlabel('Uncertainty (Std Dev)')
        plt.ylabel('Frequency')
        plt.savefig(save_dir / 'uncertainty_distribution.png')
        plt.close()

        pd.DataFrame([uncertainty_metrics]).to_csv(save_dir / 'uncertainty_metrics.csv', index=False)

        return uncertainty_metrics

    except Exception as e:
        logger.error(f"Error analyzing model uncertainty: {e}", exc_info=True)
        raise

def main():
    """Main function to run the uncertainty analysis pipeline."""
    logger.info("Starting uncertainty analysis pipeline...")
    try:
        results_dir = RESULTS_DIR / 'uncertainty_analysis_pytorch'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # --- AD/CN Model Uncertainty ---
        logger.info("Analyzing AD/CN model uncertainty...")
        ad_cn_results_dir = results_dir / 'ad_cn'
        ad_cn_results_dir.mkdir(parents=True, exist_ok=True)
        
        ad_cn_model = create_3d_cnn_model() # Create a model instance
        ad_cn_model = load_trained_model(SAVED_MODELS_DIR / 'ad_cn_final.pth', ad_cn_model)
        
        ad_cn_test_data = load_pickle(PICKLE_DIR / 'ad_cn_test.pkl')
        ad_cn_test_loader = create_data_loader(ad_cn_test_data['images'], ad_cn_test_data['labels'], batch_size=BATCH_SIZE, shuffle=False)
        
        analyze_model_uncertainty(
            model=ad_cn_model,
            test_loader=ad_cn_test_loader,
            test_labels=ad_cn_test_data['labels'],
            save_dir=ad_cn_results_dir
        )

        # --- MCI Conversion Model Uncertainty ---
        logger.info("Analyzing MCI conversion model uncertainty...")
        mci_results_dir = results_dir / 'mci_conversion'
        mci_results_dir.mkdir(parents=True, exist_ok=True)

        mci_model = create_3d_cnn_model()
        # Note: Assuming a single best MCI model is saved. If k-fold, this needs adjustment.
        # Here we load just one model for demonstration.
        try:
            mci_model = load_trained_model(SAVED_MODELS_DIR / 'mci_conversion_final_fold_1.pth', mci_model)
            mci_test_data = load_pickle(PICKLE_DIR / 'mci_conversion_test.pkl')
            mci_test_loader = create_data_loader(mci_test_data['images'], mci_test_data['labels'], batch_size=BATCH_SIZE, shuffle=False)
            
            analyze_model_uncertainty(
                model=mci_model,
                test_loader=mci_test_loader,
                test_labels=mci_test_data['labels'],
                save_dir=mci_results_dir
            )
        except FileNotFoundError:
            logger.warning("Could not find MCI conversion model, skipping uncertainty analysis for it.")

    except Exception as e:
        logger.error(f"Error in uncertainty analysis pipeline: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Uncertainty analysis pipeline completed.")

if __name__ == "__main__":
    main() 