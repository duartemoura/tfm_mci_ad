"""
Visualization utilities for plotting and visualizing results.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import nibabel as nib

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.config import TENSORBOARD_DIR, RESULTS_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_training_history(history: Dict[str, List[float]],
                         save_path: Union[str, Path]) -> None:
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Training history dictionary
        save_path: Path where to save the plot
    """
    try:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(history['accuracy'], label='Training Accuracy')
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(str(save_path))
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting training history: {str(e)}")
        raise

def plot_confusion_matrix(cm: np.ndarray,
                         class_names: List[str],
                         save_path: Union[str, Path]) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path where to save the plot
    """
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(str(save_path))
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")
        raise

def plot_roc_curve(fpr: np.ndarray,
                  tpr: np.ndarray,
                  roc_auc: float,
                  save_path: Union[str, Path]) -> None:
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        roc_auc: Area under the ROC curve
        save_path: Path where to save the plot
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(str(save_path))
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {str(e)}")
        raise

def plot_precision_recall_curve(precision: np.ndarray,
                              recall: np.ndarray,
                              avg_precision: float,
                              save_path: Union[str, Path]) -> None:
    """
    Plot precision-recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        avg_precision: Average precision score
        save_path: Path where to save the plot
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(str(save_path))
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting precision-recall curve: {str(e)}")
        raise

def plot_feature_importance(heatmap: np.ndarray,
                          save_path: Union[str, Path]) -> None:
    """
    Plot feature importance heatmap.
    
    Args:
        heatmap: Feature importance heatmap
        save_path: Path where to save the plot
    """
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap, cmap='hot')
        plt.title('Feature Importance Heatmap')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.tight_layout()
        plt.savefig(str(save_path))
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting feature importance: {str(e)}")
        raise

def plot_3d_slice(image: np.ndarray,
                 slice_idx: int,
                 save_path: Union[str, Path]) -> None:
    """
    Plot a 2D slice from a 3D image.
    
    Args:
        image: 3D image array
        slice_idx: Index of the slice to plot
        save_path: Path where to save the plot
    """
    try:
        plt.figure(figsize=(10, 8))
        plt.imshow(image[slice_idx], cmap='gray')
        plt.title(f'Slice {slice_idx}')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(str(save_path))
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting 3D slice: {str(e)}")
        raise

def plot_3d_volume(image: np.ndarray,
                  save_path: Union[str, Path],
                  num_slices: int = 4) -> None:
    """
    Plot multiple slices from a 3D volume.
    
    Args:
        image: 3D image array
        save_path: Path where to save the plot
        num_slices: Number of slices to plot
    """
    try:
        # Calculate slice indices
        depth = image.shape[0]
        slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
        
        # Create subplot grid
        fig, axes = plt.subplots(1, num_slices, figsize=(20, 5))
        
        # Plot each slice
        for i, idx in enumerate(slice_indices):
            axes[i].imshow(image[idx], cmap='gray')
            axes[i].set_title(f'Slice {idx}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(str(save_path))
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting 3D volume: {str(e)}")
        raise

def plot_metrics_comparison(metrics: Dict[str, List[float]],
                          save_path: Union[str, Path]) -> None:
    """
    Plot comparison of different metrics.
    
    Args:
        metrics: Dictionary of metric names and their values
        save_path: Path where to save the plot
    """
    try:
        # Convert metrics to DataFrame
        df = pd.DataFrame(metrics)
        
        # Create box plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df)
        plt.title('Metrics Comparison')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(str(save_path))
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting metrics comparison: {str(e)}")
        raise

def plot_correlation_matrix(data: pd.DataFrame,
                          save_path: Union[str, Path]) -> None:
    """
    Plot correlation matrix of features.
    
    Args:
        data: DataFrame containing features
        save_path: Path where to save the plot
    """
    try:
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(str(save_path))
        plt.close()
    except Exception as e:
        logger.error(f"Error plotting correlation matrix: {str(e)}")
        raise 