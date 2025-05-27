"""
Train and tune the AD/CN classification model.
This script handles model training, hyperparameter tuning, and evaluation.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt

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
    LEARNING_RATE, EPOCHS, NUM_CLASSES, IMAGE_SIZE
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ADCNHyperModel(kt.HyperModel):
    """HyperModel for AD/CN classification."""
    
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
        ADCNHyperModel(),
        objective='val_accuracy',
        max_epochs=EPOCHS,
        factor=3,
        directory=str(KERAS_TUNER_DIR),
        project_name='ad_cn_tuning'
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

def train_final_model(train_data: tf.data.Dataset,
                     val_data: tf.data.Dataset,
                     tuner: kt.Hyperband) -> tf.keras.Model:
    """
    Train the final model with best hyperparameters.
    
    Args:
        train_data: Training dataset
        val_data: Validation dataset
        tuner: Keras Tuner instance with best hyperparameters
        
    Returns:
        Trained model
    """
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Build model with best hyperparameters
    model = tuner.hypermodel.build(best_hps)
    
    # Train model
    logger.info("Training final model...")
    history = train_model(model, train_data, val_data, 'ad_cn_final')
    
    # Plot training history
    plot_training_history(history, str(TENSORBOARD_DIR / 'ad_cn_training_history.png'))
    
    return model

def evaluate_final_model(model: tf.keras.Model,
                        val_data: tf.data.Dataset) -> Dict[str, float]:
    """
    Evaluate the final model.
    
    Args:
        model: Trained model
        val_data: Validation dataset
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating final model...")
    metrics = evaluate_model(model, val_data)
    
    # Plot confusion matrix
    y_pred = model.predict(val_data)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.concatenate([y for x, y in val_data], axis=0)
    
    cm = tf.math.confusion_matrix(y_true, y_pred_classes)
    plot_confusion_matrix(cm, ['CN', 'AD'], 
                         str(TENSORBOARD_DIR / 'ad_cn_confusion_matrix.png'))
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc, 
                   str(TENSORBOARD_DIR / 'ad_cn_roc_curve.png'))
    
    return metrics

def main():
    """Main function to run the AD/CN model training pipeline."""
    logger.info("Starting AD/CN model training pipeline...")
    
    try:
        # Load preprocessed data
        data = load_pickle('ad_cn_preprocessed.pkl')
        train_data = tf.data.Dataset.from_tensor_slices(
            (data['train_images'], data['train_labels'])
        ).batch(32).prefetch(tf.data.AUTOTUNE)
        val_data = tf.data.Dataset.from_tensor_slices(
            (data['val_images'], data['val_labels'])
        ).batch(32).prefetch(tf.data.AUTOTUNE)
        
        # Tune hyperparameters
        tuner = tune_hyperparameters(train_data, val_data)
        
        # Train final model
        model = train_final_model(train_data, val_data, tuner)
        
        # Evaluate model
        metrics = evaluate_final_model(model, val_data)
        logger.info("Final model metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"Error in AD/CN model training pipeline: {str(e)}")
        sys.exit(1)
    
    logger.info("AD/CN model training pipeline completed")

if __name__ == "__main__":
    main() 