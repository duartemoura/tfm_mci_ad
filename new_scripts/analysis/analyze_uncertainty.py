"""
Analyze model uncertainty using Monte Carlo dropout.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.data_utils import load_pickle, load_csv
from scripts.utils.model_utils import (
    load_model, monte_carlo_predictions, analyze_prediction_uncertainty
)
from scripts.utils.visualization_utils import plot_metrics_comparison
from scripts.utils.config import RESULTS_DIR, SAVED_MODELS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_model_uncertainty(model_path: Path,
                            test_data: np.ndarray,
                            test_labels: np.ndarray,
                            n_samples: int = 100,
                            save_dir: Path = None) -> Dict[str, Any]:
    """
    Analyze model uncertainty using Monte Carlo dropout.
    
    Args:
        model_path: Path to the saved model
        test_data: Test data
        test_labels: Test labels
        n_samples: Number of Monte Carlo samples
        save_dir: Directory to save analysis results
        
    Returns:
        Dictionary containing uncertainty analysis results
    """
    try:
        # Load model
        model = load_model(model_path)
        
        # Generate Monte Carlo predictions
        mean_preds, std_preds = monte_carlo_predictions(
            model, test_data, n_samples=n_samples
        )
        
        # Calculate uncertainty metrics
        uncertainty = std_preds
        uncertainty_metrics = {
            'mean_uncertainty': np.mean(uncertainty),
            'std_uncertainty': np.std(uncertainty),
            'max_uncertainty': np.max(uncertainty),
            'min_uncertainty': np.min(uncertainty)
        }
        
        # Analyze prediction uncertainty
        analyze_prediction_uncertainty(
            mean_preds, std_preds, test_labels, save_dir
        )
        
        # Calculate correlation between uncertainty and prediction error
        prediction_errors = np.abs(mean_preds - test_labels)
        uncertainty_correlation = np.corrcoef(uncertainty, prediction_errors)[0, 1]
        uncertainty_metrics['uncertainty_error_correlation'] = uncertainty_correlation
        
        # Plot uncertainty distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(uncertainty, bins=30)
        plt.title('Distribution of Prediction Uncertainty')
        plt.xlabel('Uncertainty (Standard Deviation)')
        plt.ylabel('Count')
        plt.savefig(str(save_dir / 'uncertainty_distribution.png'))
        plt.close()
        
        # Plot uncertainty vs prediction error
        plt.figure(figsize=(10, 6))
        plt.scatter(uncertainty, prediction_errors, alpha=0.5)
        plt.title('Uncertainty vs Prediction Error')
        plt.xlabel('Uncertainty (Standard Deviation)')
        plt.ylabel('Absolute Prediction Error')
        plt.savefig(str(save_dir / 'uncertainty_vs_error.png'))
        plt.close()
        
        # Save uncertainty metrics
        pd.DataFrame([uncertainty_metrics]).to_csv(
            save_dir / 'uncertainty_metrics.csv', index=False
        )
        
        return uncertainty_metrics
        
    except Exception as e:
        logger.error(f"Error analyzing model uncertainty: {str(e)}")
        raise

def analyze_uncertainty_by_class(mean_preds: np.ndarray,
                               std_preds: np.ndarray,
                               test_labels: np.ndarray,
                               save_dir: Path) -> None:
    """
    Analyze uncertainty patterns by class.
    
    Args:
        mean_preds: Mean predictions from Monte Carlo sampling
        std_preds: Standard deviation of predictions
        test_labels: True labels
        save_dir: Directory to save analysis results
    """
    try:
        # Convert predictions to binary classes
        pred_classes = (mean_preds > 0.5).astype(int)
        
        # Calculate uncertainty by class
        uncertainty_by_class = pd.DataFrame({
            'true_class': test_labels,
            'pred_class': pred_classes,
            'uncertainty': std_preds
        })
        
        # Calculate statistics by class
        stats_by_class = uncertainty_by_class.groupby('true_class')['uncertainty'].agg([
            'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        # Perform t-test between classes
        class_0_uncertainty = uncertainty_by_class[uncertainty_by_class['true_class'] == 0]['uncertainty']
        class_1_uncertainty = uncertainty_by_class[uncertainty_by_class['true_class'] == 1]['uncertainty']
        t_stat, p_value = stats.ttest_ind(class_0_uncertainty, class_1_uncertainty)
        
        # Add t-test results
        stats_by_class['t_statistic'] = t_stat
        stats_by_class['p_value'] = p_value
        
        # Save statistics
        stats_by_class.to_csv(save_dir / 'uncertainty_by_class_stats.csv', index=False)
        
        # Plot uncertainty by class
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=uncertainty_by_class, x='true_class', y='uncertainty')
        plt.title('Uncertainty Distribution by True Class')
        plt.xlabel('True Class')
        plt.ylabel('Uncertainty (Standard Deviation)')
        plt.savefig(str(save_dir / 'uncertainty_by_class.png'))
        plt.close()
        
    except Exception as e:
        logger.error(f"Error analyzing uncertainty by class: {str(e)}")
        raise

def main():
    """Main function to run the uncertainty analysis pipeline."""
    logger.info("Starting uncertainty analysis pipeline...")
    
    try:
        # Create results directory
        results_dir = RESULTS_DIR / 'uncertainty_analysis'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test data
        test_data = load_pickle('test_data.pkl')
        test_labels = load_pickle('test_labels.pkl')
        
        # Analyze uncertainty for AD/CN model
        ad_cn_model_path = SAVED_MODELS_DIR / 'ad_cn_model'
        ad_cn_results_dir = results_dir / 'ad_cn'
        ad_cn_results_dir.mkdir(parents=True, exist_ok=True)
        
        ad_cn_uncertainty = analyze_model_uncertainty(
            ad_cn_model_path, test_data, test_labels,
            save_dir=ad_cn_results_dir
        )
        
        # Analyze uncertainty for MCI conversion model
        mci_model_path = SAVED_MODELS_DIR / 'mci_conversion_model'
        mci_results_dir = results_dir / 'mci_conversion'
        mci_results_dir.mkdir(parents=True, exist_ok=True)
        
        mci_uncertainty = analyze_model_uncertainty(
            mci_model_path, test_data, test_labels,
            save_dir=mci_results_dir
        )
        
        # Compare uncertainty metrics between models
        comparison_df = pd.DataFrame({
            'Model': ['AD/CN', 'MCI Conversion'],
            'Mean Uncertainty': [ad_cn_uncertainty['mean_uncertainty'],
                               mci_uncertainty['mean_uncertainty']],
            'Std Uncertainty': [ad_cn_uncertainty['std_uncertainty'],
                              mci_uncertainty['std_uncertainty']],
            'Uncertainty-Error Correlation': [
                ad_cn_uncertainty['uncertainty_error_correlation'],
                mci_uncertainty['uncertainty_error_correlation']
            ]
        })
        
        comparison_df.to_csv(results_dir / 'model_uncertainty_comparison.csv',
                           index=False)
        
    except Exception as e:
        logger.error(f"Error in uncertainty analysis pipeline: {str(e)}")
        sys.exit(1)
    
    logger.info("Uncertainty analysis pipeline completed")

if __name__ == "__main__":
    main() 