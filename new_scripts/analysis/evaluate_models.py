"""
Evaluate and analyze the trained models.
This script handles model evaluation, performance analysis, and visualization.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.data_utils import load_pickle
from scripts.utils.visualization_utils import (
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,
    plot_feature_importance
)
from scripts.utils.config import (
    SAVED_MODELS_DIR, TENSORBOARD_DIR, RESULTS_DIR,
    NUM_CLASSES, N_SPLITS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a trained model.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    return keras.models.load_model(model_path)

def evaluate_model_performance(model: tf.keras.Model,
                             test_data: tf.data.Dataset) -> Dict[str, Any]:
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained model
        test_data: Test dataset
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(test_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.concatenate([y for x, y in test_data], axis=0)
    
    # Calculate metrics
    metrics = {}
    
    # Classification report
    report = classification_report(y_true, y_pred_classes, output_dict=True)
    metrics.update(report)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    metrics['roc_auc'] = roc_auc
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred[:, 1])
    avg_precision = average_precision_score(y_true, y_pred[:, 1])
    metrics['avg_precision'] = avg_precision
    
    return metrics, (fpr, tpr), (precision, recall)

def analyze_feature_importance(model: tf.keras.Model,
                             test_data: tf.data.Dataset) -> np.ndarray:
    """
    Analyze feature importance using Grad-CAM.
    
    Args:
        model: Trained model
        test_data: Test dataset
        
    Returns:
        Feature importance maps
    """
    # Get the last convolutional layer
    last_conv_layer = model.get_layer('conv3d_2')
    
    # Create a model that outputs the last conv layer and the predictions
    grad_model = keras.Model(
        [model.inputs],
        [last_conv_layer.output, model.output]
    )
    
    # Get a batch of test images
    test_images = next(iter(test_data))[0]
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(test_images)
        loss = predictions[:, 1]  # For binary classification
    
    # Get gradients
    grads = tape.gradient(loss, conv_output)
    
    # Compute feature importance
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))
    
    # Weight the channels by their importance
    feature_maps = conv_output.numpy()
    weighted_feature_maps = np.zeros_like(feature_maps)
    
    for i in range(feature_maps.shape[-1]):
        weighted_feature_maps[:, :, :, :, i] = feature_maps[:, :, :, :, i] * pooled_grads[i]
    
    # Average the weighted feature maps
    heatmap = np.mean(weighted_feature_maps, axis=-1)
    
    return heatmap

def save_evaluation_results(metrics: Dict[str, Any],
                          model_name: str,
                          fold: int = None) -> None:
    """
    Save evaluation results to a CSV file.
    
    Args:
        metrics: Dictionary of evaluation metrics
        model_name: Name of the model
        fold: Fold number (for k-fold cross validation)
    """
    # Create results directory if it doesn't exist
    results_dir = RESULTS_DIR / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame([metrics])
    
    # Save to CSV
    filename = f'evaluation_metrics_fold_{fold}.csv' if fold else 'evaluation_metrics.csv'
    metrics_df.to_csv(results_dir / filename, index=False)

def plot_evaluation_curves(fpr: np.ndarray,
                         tpr: np.ndarray,
                         precision: np.ndarray,
                         recall: np.ndarray,
                         model_name: str,
                         fold: int = None) -> None:
    """
    Plot evaluation curves (ROC and Precision-Recall).
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        precision: Precision values
        recall: Recall values
        model_name: Name of the model
        fold: Fold number (for k-fold cross validation)
    """
    # Create results directory if it doesn't exist
    results_dir = RESULTS_DIR / model_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc(fpr, tpr):.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    
    # Save ROC curve
    filename = f'roc_curve_fold_{fold}.png' if fold else 'roc_curve.png'
    plt.savefig(results_dir / filename)
    plt.close()
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'PR curve (AP = {average_precision_score(y_true, y_pred[:, 1]):.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    # Save Precision-Recall curve
    filename = f'pr_curve_fold_{fold}.png' if fold else 'pr_curve.png'
    plt.savefig(results_dir / filename)
    plt.close()

def evaluate_ad_cn_model():
    """Evaluate the AD/CN classification model."""
    logger.info("Evaluating AD/CN classification model...")
    
    try:
        # Load model and test data
        model = load_model(str(SAVED_MODELS_DIR / 'ad_cn_model.h5'))
        test_data = load_pickle('ad_cn_test.pkl')
        test_data = tf.data.Dataset.from_tensor_slices(
            (test_data['images'], test_data['labels'])
        ).batch(32).prefetch(tf.data.AUTOTUNE)
        
        # Evaluate model performance
        metrics, (fpr, tpr), (precision, recall) = evaluate_model_performance(model, test_data)
        
        # Save evaluation results
        save_evaluation_results(metrics, 'ad_cn_model')
        
        # Plot evaluation curves
        plot_evaluation_curves(fpr, tpr, precision, recall, 'ad_cn_model')
        
        # Analyze feature importance
        heatmap = analyze_feature_importance(model, test_data)
        plot_feature_importance(heatmap, str(RESULTS_DIR / 'ad_cn_model' / 'feature_importance.png'))
        
    except Exception as e:
        logger.error(f"Error evaluating AD/CN model: {str(e)}")
        sys.exit(1)

def evaluate_mci_conversion_model():
    """Evaluate the MCI conversion prediction model."""
    logger.info("Evaluating MCI conversion prediction model...")
    
    try:
        # Load test data
        test_data = load_pickle('mci_conversion_test.pkl')
        test_data = tf.data.Dataset.from_tensor_slices(
            (test_data['images'], test_data['labels'])
        ).batch(32).prefetch(tf.data.AUTOTUNE)
        
        # Evaluate each fold
        all_metrics = []
        all_fpr_tpr = []
        all_precision_recall = []
        
        for fold in range(N_SPLITS):
            # Load model
            model = load_model(str(SAVED_MODELS_DIR / f'mci_conversion_fold_{fold + 1}.h5'))
            
            # Evaluate model performance
            metrics, (fpr, tpr), (precision, recall) = evaluate_model_performance(model, test_data)
            
            # Save evaluation results
            save_evaluation_results(metrics, 'mci_conversion_model', fold + 1)
            
            # Plot evaluation curves
            plot_evaluation_curves(fpr, tpr, precision, recall, 'mci_conversion_model', fold + 1)
            
            # Store results
            all_metrics.append(metrics)
            all_fpr_tpr.append((fpr, tpr))
            all_precision_recall.append((precision, recall))
        
        # Calculate average metrics
        avg_metrics = {
            metric: np.mean([m[metric] for m in all_metrics])
            for metric in all_metrics[0].keys()
        }
        
        # Save average metrics
        save_evaluation_results(avg_metrics, 'mci_conversion_model', 'average')
        
        # Plot average curves
        avg_fpr = np.mean([fpr for fpr, _ in all_fpr_tpr], axis=0)
        avg_tpr = np.mean([tpr for _, tpr in all_fpr_tpr], axis=0)
        avg_precision = np.mean([precision for precision, _ in all_precision_recall], axis=0)
        avg_recall = np.mean([recall for _, recall in all_precision_recall], axis=0)
        
        plot_evaluation_curves(avg_fpr, avg_tpr, avg_precision, avg_recall, 'mci_conversion_model', 'average')
        
    except Exception as e:
        logger.error(f"Error evaluating MCI conversion model: {str(e)}")
        sys.exit(1)

def main():
    """Main function to run the model evaluation pipeline."""
    logger.info("Starting model evaluation pipeline...")
    
    try:
        # Evaluate AD/CN model
        evaluate_ad_cn_model()
        
        # Evaluate MCI conversion model
        evaluate_mci_conversion_model()
        
    except Exception as e:
        logger.error(f"Error in model evaluation pipeline: {str(e)}")
        sys.exit(1)
    
    logger.info("Model evaluation pipeline completed")

if __name__ == "__main__":
    main() 