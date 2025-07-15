"""
Train and tune the AD/CN classification model using PyTorch and Optuna.
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
import optuna
from sklearn.metrics import confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from new_scripts_pytorch.utils.data_utils import load_pickle, create_data_loader
from new_scripts_pytorch.utils.model_utils import (
    create_3d_cnn_model, train_model, evaluate_model, load_trained_model
)
from new_scripts_pytorch.utils.visualization_utils import (
    plot_training_history, plot_confusion_matrix, plot_roc_curve
)
from new_scripts_pytorch.utils.config import (
    SAVED_MODELS_DIR, TENSORBOARD_DIR,
    LEARNING_RATE, EPOCHS, NUM_CLASSES, IMAGE_SIZE, BATCH_SIZE, PICKLE_DIR
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def objective(trial: optuna.trial.Trial, train_loader, val_loader) -> float:
    """
    Optuna objective function for hyperparameter tuning.
    """
    # Hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    filters1 = trial.suggest_categorical('filters1', [16, 32, 64])
    filters2 = trial.suggest_categorical('filters2', [32, 64, 128])
    filters3 = trial.suggest_categorical('filters3', [64, 128, 256])
    dense_units = trial.suggest_categorical('dense_units', [256, 512, 1024])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)

    filters = [filters1, filters2, filters3]
    
    # Create model
    model = create_3d_cnn_model(
        filters=filters,
        dense_units=dense_units,
        dropout_rate=dropout_rate
    )
    
    # Train model
    history = train_model(
        model,
        train_loader,
        val_loader,
        f'ad_cn_tuning_trial_{trial.number}',
        learning_rate=lr,
        epochs=EPOCHS
    )
    
    # Return validation accuracy
    return max(history['val_acc'])

def tune_hyperparameters(train_loader, val_loader) -> optuna.study.Study:
    """
    Tune model hyperparameters using Optuna.
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, train_loader, val_loader), n_trials=50)
    
    logger.info("Best hyperparameters:")
    for param, value in study.best_params.items():
        logger.info(f"{param}: {value}")
        
    return study

def train_final_model(train_loader, val_loader, best_params: Dict[str, Any]) -> nn.Module:
    """
    Train the final model with best hyperparameters.
    """
    model = create_3d_cnn_model(
        filters=[best_params['filters1'], best_params['filters2'], best_params['filters3']],
        dense_units=best_params['dense_units'],
        dropout_rate=best_params['dropout_rate']
    )
    
    history = train_model(
        model,
        train_loader,
        val_loader,
        'ad_cn_final',
        learning_rate=best_params['lr'],
        epochs=EPOCHS
    )
    
    plot_training_history(history, str(TENSORBOARD_DIR / 'ad_cn_training_history.png'))
    
    return model
    
def evaluate_final_model(model: nn.Module, val_loader: DataLoader):
    """
    Evaluate the final model.
    """
    logger.info("Evaluating final model...")
    metrics, y_true, y_pred = evaluate_model(model, val_loader)
    
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, ['CN', 'AD'], 
                         str(TENSORBOARD_DIR / 'ad_cn_confusion_matrix.png'))
    
    # Get probabilities for ROC curve
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    y_prob = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_prob.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    
    y_prob = np.array(y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, 
                   str(TENSORBOARD_DIR / 'ad_cn_roc_curve.png'))
                   
    return metrics

def main():
    """Main function to run the AD/CN model training pipeline."""
    logger.info("Starting AD/CN model training pipeline...")
    
    try:
        # Load preprocessed data
        data = load_pickle(PICKLE_DIR / 'ad_cn_preprocessed.pkl')
        
        train_loader = create_data_loader(
            data['train_images'], data['train_labels'], batch_size=BATCH_SIZE
        )
        val_loader = create_data_loader(
            data['val_images'], data['val_labels'], batch_size=BATCH_SIZE, shuffle=False
        )
        
        # Tune hyperparameters
        study = tune_hyperparameters(train_loader, val_loader)
        
        # Train final model
        model = train_final_model(train_loader, val_loader, study.best_params)
        
        # Evaluate model
        metrics = evaluate_final_model(model, val_loader)
        logger.info("Final model metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"Error in AD/CN model training pipeline: {str(e)}")
        sys.exit(1)
        
    logger.info("AD/CN model training pipeline completed")

if __name__ == "__main__":
    main() 