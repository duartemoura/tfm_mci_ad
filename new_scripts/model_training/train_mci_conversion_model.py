"""
Train and tune the MCI conversion prediction model.
This script handles model training, hyperparameter tuning, and evaluation.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from sklearn.model_selection import KFold

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.data_utils import load_pickle
from scripts.utils.model_utils import (
    create_3d_cnn_model, train_model, evaluate_model
)
from scripts.utils.visualization_utils import (
    plot_training_history, plot_confusion_matrix, plot_roc_curve
)
from scripts.utils.config import (
    KERAS_TUNER_DIR, SAVED_MODELS_DIR, TENSORBOARD_DIR,
    LEARNING_RATE, EPOCHS, NUM_CLASSES, IMAGE_SIZE, N_SPLITS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MCIConversionHyperModel(kt.HyperModel):
    """HyperModel for MCI conversion prediction."""
    
    def build(self, hp):
        """Build the model with hyperparameters."""
        model = keras.Sequential()
        
        # First Convolutional Block
        filters = hp.Int('filters_1', min_value=16, max_value=64, step=16)
        model.add(keras.layers.Conv3D(filters, (3, 3, 3), activation='relu', 
                                    padding='same', input_shape=IMAGE_SIZE + (1,)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling3D((2, 2, 2)))
        
        # Second Convolutional Block
        filters = hp.Int('filters_2', min_value=32, max_value=128, step=32)
        model.add(keras.layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling3D((2, 2, 2)))
        
        # Third Convolutional Block
        filters = hp.Int('filters_3', min_value=64, max_value=256, step=64)
        model.add(keras.layers.Conv3D(filters, (3, 3, 3), activation='relu', padding='same'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling3D((2, 2, 2)))
        
        # Dense Layers
        model.add(keras.layers.Flatten())
        
        # Dense layer units
        dense_units = hp.Int('dense_units', min_value=256, max_value=1024, step=256)
        model.add(keras.layers.Dense(dense_units, activation='relu'))
        
        # Dropout rate
        dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
        model.add(keras.layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
        
        # Compile model
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

def prepare_kfold_data(data: Dict[str, np.ndarray]) -> Tuple[list, list]:
    """
    Prepare data for k-fold cross validation.
    
    Args:
        data: Dictionary containing images and labels
        
    Returns:
        Tuple of lists containing k-fold train and validation datasets
    """
    kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    train_datasets = []
    val_datasets = []
    
    for train_idx, val_idx in kfold.split(data['images']):
        # Create train dataset
        train_images = data['images'][train_idx]
        train_labels = data['labels'][train_idx]
        train_data = tf.data.Dataset.from_tensor_slices(
            (train_images, train_labels)
        ).batch(32).prefetch(tf.data.AUTOTUNE)
        train_datasets.append(train_data)
        
        # Create validation dataset
        val_images = data['images'][val_idx]
        val_labels = data['labels'][val_idx]
        val_data = tf.data.Dataset.from_tensor_slices(
            (val_images, val_labels)
        ).batch(32).prefetch(tf.data.AUTOTUNE)
        val_datasets.append(val_data)
    
    return train_datasets, val_datasets

def tune_hyperparameters(train_data: tf.data.Dataset, 
                        val_data: tf.data.Dataset) -> kt.Hyperband:
    """
    Tune model hyperparameters using Keras Tuner.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        
    Returns:
        Tuned hyperparameters
    """
    tuner = kt.Hyperband(
        MCIConversionHyperModel(),
        objective='val_accuracy',
        max_epochs=EPOCHS,
        factor=3,
        directory=str(KERAS_TUNER_DIR),
        project_name='mci_conversion_tuning'
    )
    
    # Define early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Start hyperparameter tuning
    logger.info("Starting hyperparameter tuning...")
    tuner.search(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=[early_stopping]
    )
    
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    logger.info("Best hyperparameters:")
    for param, value in best_hps.values.items():
        logger.info(f"{param}: {value}")
    
    return tuner

def train_fold_model(train_data: tf.data.Dataset,
                    val_data: tf.data.Dataset,
                    tuner: kt.Hyperband,
                    fold: int) -> Tuple[tf.keras.Model, Dict[str, float]]:
    """
    Train model for a specific fold.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        tuner: Keras Tuner instance with best hyperparameters
        fold: Fold number
        
    Returns:
        Tuple of trained model and evaluation metrics
    """
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Build model with best hyperparameters
    model = tuner.hypermodel.build(best_hps)
    
    # Train model
    logger.info(f"Training model for fold {fold}...")
    history = train_model(model, train_data, val_data, f'mci_conversion_fold_{fold}')
    
    # Plot training history
    plot_training_history(
        history, 
        str(TENSORBOARD_DIR / f'mci_conversion_fold_{fold}_training_history.png')
    )
    
    # Evaluate model
    metrics = evaluate_model(model, val_data)
    
    # Plot confusion matrix
    y_pred = model.predict(val_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.concatenate([y for x, y in val_data], axis=0)
    
    cm = tf.math.confusion_matrix(y_true, y_pred_classes)
    plot_confusion_matrix(
        cm, 
        ['Stable MCI', 'Converter MCI'],
        str(TENSORBOARD_DIR / f'mci_conversion_fold_{fold}_confusion_matrix.png')
    )
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(
        fpr, tpr, roc_auc,
        str(TENSORBOARD_DIR / f'mci_conversion_fold_{fold}_roc_curve.png')
    )
    
    return model, metrics

def main():
    """Main function to run the MCI conversion model training pipeline."""
    logger.info("Starting MCI conversion model training pipeline...")
    
    try:
        # Load preprocessed data
        data = load_pickle('mci_conversion_preprocessed.pkl')
        
        # Prepare k-fold datasets
        train_datasets, val_datasets = prepare_kfold_data(data)
        
        # Train and evaluate models for each fold
        fold_metrics = []
        for fold in range(N_SPLITS):
            logger.info(f"Processing fold {fold + 1}/{N_SPLITS}")
            
            # Tune hyperparameters
            tuner = tune_hyperparameters(train_datasets[fold], val_datasets[fold])
            
            # Train and evaluate model
            model, metrics = train_fold_model(
                train_datasets[fold], 
                val_datasets[fold], 
                tuner,
                fold + 1
            )
            fold_metrics.append(metrics)
            
            # Save model
            model.save(str(SAVED_MODELS_DIR / f'mci_conversion_fold_{fold + 1}.h5'))
        
        # Calculate and log average metrics
        avg_metrics = {
            metric: np.mean([m[metric] for m in fold_metrics])
            for metric in fold_metrics[0].keys()
        }
        logger.info("Average metrics across folds:")
        for metric, value in avg_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"Error in MCI conversion model training pipeline: {str(e)}")
        sys.exit(1)
    
    logger.info("MCI conversion model training pipeline completed")

if __name__ == "__main__":
    main() 