"""
Visualization utilities for the PyTorch port.
The implementation is identical to the TensorFlow version but imports the
PyTorch configuration module so that output files land in the *_pytorch* result
folders.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc

from .config import TENSORBOARD_DIR, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Plots (functions copied verbatim from the original file)
# -----------------------------------------------------------------------------

def plot_training_history(history: Dict[str, List[float]], save_path: Union[str, Path]) -> None:
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(history['train_loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(history['train_acc'], label='Training Acc')
        ax2.plot(history['val_acc'], label='Validation Acc')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        logger.error("Error plotting training history: %s", e)
        raise


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Union[str, Path]) -> None:
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        logger.error("Error plotting confusion matrix: %s", e)
        raise


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, save_path: Union[str, Path]) -> None:
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
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        logger.error("Error plotting ROC curve: %s", e)
        raise 