"""
Model utilities for creating, training, and evaluating models.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Tuple, List, Union
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.config import (
    SAVED_MODELS_DIR, TENSORBOARD_DIR, LEARNING_RATE,
    EPOCHS, NUM_CLASSES, IMAGE_SIZE
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_3d_cnn_model(input_shape: Tuple[int, int, int, int] = None,
                       num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Create a 3D CNN model for classification.
    
    Args:
        input_shape: Input shape (height, width, depth, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    try:
        if input_shape is None:
            input_shape = IMAGE_SIZE + (1,)
        
        model = keras.Sequential([
            # First Convolutional Block
            keras.layers.Conv3D(32, (3, 3, 3), activation='relu',
                              padding='same', input_shape=input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling3D((2, 2, 2)),
            
            # Second Convolutional Block
            keras.layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling3D((2, 2, 2)),
            
            # Third Convolutional Block
            keras.layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling3D((2, 2, 2)),
            
            # Dense Layers
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    except Exception as e:
        logger.error(f"Error creating 3D CNN model: {str(e)}")
        raise

def train_model(model: tf.keras.Model,
                train_data: tf.data.Dataset,
                val_data: tf.data.Dataset,
                model_name: str) -> Dict[str, List[float]]:
    """
    Train a model with callbacks for monitoring and early stopping.
    
    Args:
        model: Model to train
        train_data: Training dataset
        val_data: Validation dataset
        model_name: Name of the model (for saving checkpoints and logs)
        
    Returns:
        Training history
    """
    try:
        # Create callbacks
        callbacks = [
            # Model checkpoint
            keras.callbacks.ModelCheckpoint(
                filepath=str(SAVED_MODELS_DIR / f'{model_name}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            # TensorBoard
            keras.callbacks.TensorBoard(
                log_dir=str(TENSORBOARD_DIR / model_name),
                histogram_freq=1
            ),
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=EPOCHS,
            callbacks=callbacks
        )
        
        return history.history
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def evaluate_model(model: tf.keras.Model,
                  test_data: tf.data.Dataset) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        test_data: Test dataset
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(test_data)
        
        # Get predictions
        y_pred = model.predict(test_data)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.concatenate([y for x, y in test_data], axis=0)
        
        # Calculate metrics
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
        
        # Add per-class metrics
        for i in range(NUM_CLASSES):
            class_mask = (y_true == i)
            class_pred = y_pred_classes[class_mask]
            class_true = y_true[class_mask]
            
            if len(class_true) > 0:
                class_accuracy = np.mean(class_pred == class_true)
                metrics[f'class_{i}_accuracy'] = class_accuracy
        
        return metrics
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def load_trained_model(model_path: str) -> tf.keras.Model:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    try:
        return keras.models.load_model(model_path)
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        raise

def save_model_summary(model: tf.keras.Model,
                      file_path: str) -> None:
    """
    Save model summary to a text file.
    
    Args:
        model: Model to summarize
        file_path: Path where to save the summary
    """
    try:
        # Get model summary
        summary_list = []
        model.summary(print_fn=lambda x: summary_list.append(x))
        summary = '\n'.join(summary_list)
        
        # Save to file
        with open(file_path, 'w') as f:
            f.write(summary)
    except Exception as e:
        logger.error(f"Error saving model summary: {str(e)}")
        raise

def get_model_weights(model: tf.keras.Model) -> Dict[str, np.ndarray]:
    """
    Get model weights for analysis.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary of layer names and their weights
    """
    try:
        weights = {}
        for layer in model.layers:
            if layer.weights:
                weights[layer.name] = [w.numpy() for w in layer.weights]
        return weights
    except Exception as e:
        logger.error(f"Error getting model weights: {str(e)}")
        raise

def analyze_model_weights(weights: Dict[str, List[np.ndarray]]) -> None:
    """
    Analyze and visualize model weights.
    
    Args:
        weights: Dictionary of layer names and their weights
    """
    try:
        for layer_name, layer_weights in weights.items():
            for i, weight in enumerate(layer_weights):
                # Plot weight distribution
                plt.figure(figsize=(10, 6))
                sns.histplot(weight.flatten(), bins=50)
                plt.title(f'{layer_name} - Weight {i} Distribution')
                plt.xlabel('Weight Value')
                plt.ylabel('Count')
                
                # Save plot
                plot_path = TENSORBOARD_DIR / f'{layer_name}_weight_{i}_distribution.png'
                plt.savefig(str(plot_path))
                plt.close()
    except Exception as e:
        logger.error(f"Error analyzing model weights: {str(e)}")
        raise

def monte_carlo_predictions(model: tf.keras.Model,
                          data: np.ndarray,
                          num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Monte Carlo predictions using dropout at inference time.
    
    Args:
        model: Trained model with dropout layers
        data: Input data
        num_samples: Number of Monte Carlo samples
        
    Returns:
        Tuple of (mean predictions, standard deviation of predictions)
    """
    try:
        predictions = []
        
        for _ in range(num_samples):
            pred = model(data, training=True)  # Enable dropout
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred
    except Exception as e:
        logger.error(f"Error generating Monte Carlo predictions: {str(e)}")
        raise

def analyze_prediction_uncertainty(mean_pred: np.ndarray,
                                 std_pred: np.ndarray,
                                 true_labels: np.ndarray,
                                 save_path: Union[str, Path]) -> None:
    """
    Analyze and visualize prediction uncertainty.
    
    Args:
        mean_pred: Mean predictions
        std_pred: Standard deviation of predictions
        true_labels: True labels
        save_path: Path where to save the analysis plots
    """
    try:
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Sort by true label
        sort_idx = np.argsort(true_labels)
        mean_pred = mean_pred[sort_idx]
        std_pred = std_pred[sort_idx]
        true_labels = true_labels[sort_idx]
        
        # Plot predictions with uncertainty
        plt.errorbar(range(len(true_labels)), mean_pred,
                    yerr=std_pred, fmt='o', alpha=0.5,
                    label='Predictions with Uncertainty')
        plt.plot(true_labels, 'r-', label='True Labels')
        
        plt.title('Monte Carlo Predictions with Uncertainty')
        plt.xlabel('Sample Index')
        plt.ylabel('Prediction')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(str(save_path))
        plt.close()
        
        # Calculate and log uncertainty metrics
        uncertainty_metrics = {
            'mean_uncertainty': np.mean(std_pred),
            'max_uncertainty': np.max(std_pred),
            'min_uncertainty': np.min(std_pred),
            'uncertainty_std': np.std(std_pred)
        }
        
        logger.info("Uncertainty metrics:")
        for metric, value in uncertainty_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"Error analyzing prediction uncertainty: {str(e)}")
        raise 