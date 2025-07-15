"""
Model utilities for creating, training, and evaluating PyTorch models.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, Tuple, List, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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

class CNN3D(nn.Module):
    """3D CNN model for classification."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = IMAGE_SIZE,
                 num_classes: int = NUM_CLASSES,
                 filters: List[int] = [32, 64, 128],
                 dense_units: int = 512,
                 dropout_rate: float = 0.5):
        """
        Args:
            input_shape: Input shape (depth, height, width)
            num_classes: Number of output classes
            filters: List of filter sizes for convolutional layers
            dense_units: Number of units in the dense layer
            dropout_rate: Dropout rate
        """
        super(CNN3D, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(1, filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(filters[0]),
            nn.MaxPool3d(2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(filters[0], filters[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(filters[1]),
            nn.MaxPool3d(2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(filters[1], filters[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(filters[2]),
            nn.MaxPool3d(2)
        )
        
        # Calculate the flattened size
        self._flatten_size = self._get_flatten_size(input_shape)
        
        self.dense_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flatten_size, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, num_classes)
        )
        
    def _get_flatten_size(self, input_shape: Tuple[int, int, int]) -> int:
        """Helper to calculate the size of the flattened layer."""
        x = torch.zeros(1, 1, *input_shape)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x.flatten().shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Add channel dimension
        if x.dim() == 4:
            x = x.unsqueeze(1)
            
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.dense_block(x)
        return x

def create_3d_cnn_model(input_shape: Tuple[int, int, int] = IMAGE_SIZE,
                       num_classes: int = NUM_CLASSES,
                       filters: List[int] = [32, 64, 128],
                       dense_units: int = 512,
                       dropout_rate: float = 0.5) -> nn.Module:
    """
    Create a 3D CNN model.
    
    Args:
        input_shape: Input shape (depth, height, width)
        num_classes: Number of output classes
        filters: List of filter sizes for convolutional layers
        dense_units: Number of units in the dense layer
        dropout_rate: Dropout rate
        
    Returns:
        PyTorch model
    """
    try:
        model = CNN3D(input_shape, num_classes, filters, dense_units, dropout_rate)
        return model
    except Exception as e:
        logger.error(f"Error creating 3D CNN model: {str(e)}")
        raise

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                model_name: str,
                learning_rate: float = LEARNING_RATE,
                epochs: int = EPOCHS) -> Dict[str, List[float]]:
    """
    Train a PyTorch model.
    
    Args:
        model: Model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        model_name: Name for saving checkpoints and logs
        learning_rate: Learning rate
        epochs: Number of epochs
        
    Returns:
        Training history
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR / model_name))
        
        best_val_accuracy = 0.0
        epochs_no_improve = 0
        patience = 10
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = correct_train / total_train
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            
            # Validation phase
            model.eval()
            running_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            val_loss = running_loss / len(val_loader)
            val_acc = correct_val / total_val
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, "
                        f"Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
            
            # Early stopping and model checkpointing
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), str(SAVED_MODELS_DIR / f'{model_name}.pth'))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered.")
                break
                
        writer.close()
        return history
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def evaluate_model(model: nn.Module,
                  test_loader: DataLoader) -> Dict[str, float]:
    """
    Evaluate a trained PyTorch model.
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        criterion = nn.CrossEntropyLoss()
        running_loss = 0.0
        correct_test = 0
        total_test = 0
        
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
        test_loss = running_loss / len(test_loader)
        test_acc = correct_test / total_test
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_acc
        }
        
        return metrics, y_true, y_pred
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def load_trained_model(model_path: str,
                       model: nn.Module) -> nn.Module:
    """
    Load a trained PyTorch model.
    
    Args:
        model_path: Path to the saved model state dict
        model: The model architecture to load the weights into
        
    Returns:
        Loaded model
    """
    try:
        model.load_state_dict(torch.load(model_path))
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        raise

def save_model_summary(model: nn.Module,
                      input_size: tuple,
                      file_path: str):
    """
    Save model summary to a text file.
    
    Args:
        model: Model to summarize
        input_size: Input size for the model
        file_path: Path to save the summary
    """
    try:
        from torchsummary import summary
        original_stdout = sys.stdout
        with open(file_path, 'w') as f:
            sys.stdout = f
            summary(model, input_size)
            sys.stdout = original_stdout
    except ImportError:
        logger.warning("torchsummary not installed. Skipping model summary save.")
        with open(file_path, 'w') as f:
            f.write(str(model))
    except Exception as e:
        logger.error(f"Error saving model summary: {str(e)}")

def get_model_weights(model: nn.Module) -> Dict[str, np.ndarray]:
    """
    Get model weights.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary of layer names and their weights
    """
    try:
        return {name: param.cpu().detach().numpy() for name, param in model.named_parameters()}
    except Exception as e:
        logger.error(f"Error getting model weights: {str(e)}")
        raise

def monte_carlo_predictions(model: nn.Module,
                          data_loader: DataLoader,
                          num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Monte Carlo predictions with dropout enabled at inference time.
    
    Args:
        model: Model with dropout layers
        data_loader: DataLoader for the data
        num_samples: Number of forward passes
        
    Returns:
        Tuple of mean predictions and standard deviation of predictions
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Enable dropout during inference
        model.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                sample_preds = []
                for inputs, _ in data_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    sample_preds.append(torch.softmax(outputs, dim=1).cpu().numpy())
                predictions.append(np.vstack(sample_preds))
                
        predictions = np.array(predictions)
        mean_preds = np.mean(predictions, axis=0)
        std_preds = np.std(predictions, axis=0)
        
        return mean_preds, std_preds
    except Exception as e:
        logger.error(f"Error during Monte Carlo predictions: {str(e)}")
        raise 