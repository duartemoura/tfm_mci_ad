"""
Train and tune the MCI conversion prediction model using PyTorch and Optuna.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Tuple, List
import torch
import torch.nn as nn
import optuna
from sklearn.model_selection import KFold
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
    LEARNING_RATE, EPOCHS, NUM_CLASSES, IMAGE_SIZE, N_SPLITS, BATCH_SIZE, PICKLE_DIR
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def objective(trial: optuna.trial.Trial, train_loader: DataLoader, val_loader: DataLoader) -> float:
    """Optuna objective function for a single fold."""
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    filters1 = trial.suggest_categorical('filters1', [16, 32, 64])
    filters2 = trial.suggest_categorical('filters2', [32, 64, 128])
    filters3 = trial.suggest_categorical('filters3', [64, 128, 256])
    dense_units = trial.suggest_categorical('dense_units', [256, 512, 1024])
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)

    model = create_3d_cnn_model(
        filters=[filters1, filters2, filters3],
        dense_units=dense_units,
        dropout_rate=dropout_rate
    )
    
    history = train_model(
        model,
        train_loader,
        val_loader,
        f'mci_conversion_tuning_trial_{trial.number}',
        learning_rate=lr,
        epochs=EPOCHS
    )
    
    return max(history['val_acc'])

def prepare_kfold_data(data: Dict[str, np.ndarray]) -> List[Tuple[DataLoader, DataLoader]]:
    """Prepare k-fold DataLoaders."""
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_dataloaders = []
    
    for train_idx, val_idx in kfold.split(data['images']):
        train_loader = create_data_loader(
            data['images'][train_idx], data['labels'][train_idx], batch_size=BATCH_SIZE
        )
        val_loader = create_data_loader(
            data['images'][val_idx], data['labels'][val_idx], batch_size=BATCH_SIZE, shuffle=False
        )
        fold_dataloaders.append((train_loader, val_loader))
        
    return fold_dataloaders

def main():
    """Main function to run the MCI conversion model training pipeline."""
    logger.info("Starting MCI conversion model training pipeline...")
    
    try:
        data = load_pickle(PICKLE_DIR / 'mci_conversion_preprocessed.pkl')
        fold_dataloaders = prepare_kfold_data(data)
        
        all_fold_metrics = []
        for fold, (train_loader, val_loader) in enumerate(fold_dataloaders):
            logger.info(f"Processing fold {fold + 1}/{N_SPLITS}")
            
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, train_loader, val_loader), n_trials=25)
            
            logger.info(f"Best hyperparameters for fold {fold + 1}: {study.best_params}")
            
            model = create_3d_cnn_model(
                filters=[study.best_params['filters1'], study.best_params['filters2'], study.best_params['filters3']],
                dense_units=study.best_params['dense_units'],
                dropout_rate=study.best_params['dropout_rate']
            )
            
            history = train_model(
                model,
                train_loader,
                val_loader,
                f'mci_conversion_final_fold_{fold+1}',
                learning_rate=study.best_params['lr'],
                epochs=EPOCHS
            )
            
            plot_training_history(history, TENSORBOARD_DIR / f'mci_conversion_fold_{fold+1}_history.png')
            
            metrics, y_true, y_pred = evaluate_model(model, val_loader)
            all_fold_metrics.append(metrics)
            
            cm = confusion_matrix(y_true, y_pred)
            plot_confusion_matrix(cm, ['Stable', 'Converter'], TENSORBOARD_DIR / f'mci_conversion_fold_{fold+1}_cm.png')
            
            # ROC Curve
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
            plot_roc_curve(fpr, tpr, roc_auc, TENSORBOARD_DIR / f'mci_conversion_fold_{fold+1}_roc.png')

        # Average metrics across folds
        avg_metrics = pd.DataFrame(all_fold_metrics).mean().to_dict()
        logger.info("Average metrics across all folds:")
        for metric, value in avg_metrics.items():
            logger.info(f"{metric}: {value:.4f}")

    except Exception as e:
        logger.error(f"Error in MCI conversion pipeline: {e}")
        sys.exit(1)
        
    logger.info("MCI conversion model training pipeline completed")

if __name__ == "__main__":
    main() 