"""
Evaluate and analyze the trained PyTorch models.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from new_scripts_pytorch.utils.data_utils import load_pickle, create_data_loader
from new_scripts_pytorch.utils.model_utils import create_3d_cnn_model, load_trained_model
from new_scripts_pytorch.utils.visualization_utils import (
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,
    plot_feature_importance
)
from new_scripts_pytorch.utils.config import (
    SAVED_MODELS_DIR, RESULTS_DIR, NUM_CLASSES, N_SPLITS, BATCH_SIZE, PICKLE_DIR
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model_performance(model: nn.Module, test_loader: DataLoader) -> Dict[str, Any]:
    """Evaluate model performance on test data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true, y_pred_classes, y_pred_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred_classes.extend(preds.cpu().numpy())
            y_pred_probs.extend(probs.cpu().numpy())

    y_pred_probs = np.array(y_pred_probs)
    report = classification_report(y_true, y_pred_classes, output_dict=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs[:, 1])
    avg_precision = average_precision_score(y_true, y_pred_probs[:, 1])

    return {
        'report': report, 'roc_auc': roc_auc, 'avg_precision': avg_precision,
        'roc_curve': (fpr, tpr), 'pr_curve': (precision, recall)
    }

def save_evaluation_results(metrics: Dict[str, Any], model_name: str, fold: int = None):
    """Save evaluation results to a CSV file."""
    results_dir = RESULTS_DIR / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame([metrics['report']['weighted avg']])
    metrics_df['roc_auc'] = metrics['roc_auc']
    metrics_df['avg_precision'] = metrics['avg_precision']
    
    filename = f'evaluation_metrics_fold_{fold}.csv' if fold is not None else 'evaluation_metrics.csv'
    metrics_df.to_csv(results_dir / filename, index=False)

def plot_evaluation_curves(metrics: Dict[str, Any], model_name: str, fold: int = None):
    """Plot ROC and Precision-Recall curves."""
    results_dir = RESULTS_DIR / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    fpr, tpr = metrics['roc_curve']
    plot_roc_curve(fpr, tpr, metrics['roc_auc'], results_dir / (f'roc_curve_fold_{fold}.png' if fold is not None else 'roc_curve.png'))
    
    precision, recall = metrics['pr_curve']
    plot_precision_recall_curve(recall, precision, metrics['avg_precision'], results_dir / (f'pr_curve_fold_{fold}.png' if fold is not None else 'pr_curve.png'))

def evaluate_ad_cn_model():
    """Evaluate the AD/CN classification model."""
    logger.info("Evaluating AD/CN classification model...")
    try:
        model = create_3d_cnn_model()
        model = load_trained_model(SAVED_MODELS_DIR / 'ad_cn_final.pth', model)
        test_data = load_pickle(PICKLE_DIR / 'ad_cn_test.pkl')
        test_loader = create_data_loader(test_data['images'], test_data['labels'], batch_size=BATCH_SIZE, shuffle=False)
        
        metrics = evaluate_model_performance(model, test_loader)
        save_evaluation_results(metrics, 'ad_cn_model')
        plot_evaluation_curves(metrics, 'ad_cn_model')

    except Exception as e:
        logger.error(f"Error evaluating AD/CN model: {e}", exc_info=True)

def evaluate_mci_conversion_model():
    """Evaluate the MCI conversion prediction model."""
    logger.info("Evaluating MCI conversion prediction model...")
    try:
        test_data = load_pickle(PICKLE_DIR / 'mci_conversion_test.pkl')
        test_loader = create_data_loader(test_data['images'], test_data['labels'], batch_size=BATCH_SIZE, shuffle=False)
        
        all_metrics = []
        for fold in range(N_SPLITS):
            logger.info(f"Evaluating fold {fold + 1}/{N_SPLITS}")
            model = create_3d_cnn_model()
            model = load_trained_model(SAVED_MODELS_DIR / f'mci_conversion_final_fold_{fold+1}.pth', model)
            
            metrics = evaluate_model_performance(model, test_loader)
            all_metrics.append(metrics['report']['weighted avg'])
            save_evaluation_results(metrics, 'mci_conversion_model', fold=fold + 1)
            plot_evaluation_curves(metrics, 'mci_conversion_model', fold=fold + 1)
        
        avg_metrics = pd.DataFrame(all_metrics).mean().to_dict()
        logger.info(f"Average metrics across folds: {avg_metrics}")
        pd.DataFrame([avg_metrics]).to_csv(RESULTS_DIR / 'mci_conversion_model' / 'average_metrics.csv', index=False)

    except Exception as e:
        logger.error(f"Error evaluating MCI conversion model: {e}", exc_info=True)

def main():
    """Main function to run model evaluations."""
    evaluate_ad_cn_model()
    evaluate_mci_conversion_model()

if __name__ == "__main__":
    main() 