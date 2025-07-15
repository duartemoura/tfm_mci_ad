"""
Data utilities for the FDG MCI to AD analysis project.
Contains functions for loading, processing, and saving data.
"""

import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import pickle
from typing import Tuple, List, Dict, Any, Union
from sklearn.preprocessing import StandardScaler
import logging

from .config import (
    NEW_DATA_DIR, PICKLE_DIR, IMAGE_SIZE, BATCH_SIZE,
    ADNI_MERGE_PATH, DATA_DIR, PROCESSED_DATA_DIR
)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_nifti(file_path: Union[str, Path]) -> np.ndarray:
    """
    Load a NIfTI file and return its data as a numpy array.
    
    Args:
        file_path: Path to the NIfTI file
        
    Returns:
        Numpy array containing the image data
    """
    try:
        nifti_img = nib.load(str(file_path))
        return nifti_img.get_fdata()
    except Exception as e:
        logger.error(f"Error loading NIfTI file {file_path}: {str(e)}")
        raise

def save_nifti(data: np.ndarray,
              file_path: Union[str, Path],
              affine: np.ndarray = None) -> None:
    """
    Save a numpy array as a NIfTI file.
    
    Args:
        data: Numpy array to save
        file_path: Path where to save the NIfTI file
        affine: Affine transformation matrix (optional)
    """
    try:
        if affine is None:
            affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        nib.save(nifti_img, str(file_path))
    except Exception as e:
        logger.error(f"Error saving NIfTI file {file_path}: {str(e)}")
        raise

def load_pickle(file_path: Union[str, Path]) -> Any:
    """
    Load data from a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        Loaded data
    """
    try:
        with open(str(file_path), 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {str(e)}")
        raise

def save_pickle(data: Any,
               file_path: Union[str, Path]) -> None:
    """
    Save data to a pickle file.
    
    Args:
        data: Data to save
        file_path: Path where to save the pickle file
    """
    try:
        with open(str(file_path), 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.error(f"Error saving pickle file {file_path}: {str(e)}")
        raise

def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Loaded data as a pandas DataFrame
    """
    try:
        return pd.read_csv(str(file_path))
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {str(e)}")
        raise

def save_csv(data: pd.DataFrame,
            file_path: Union[str, Path]) -> None:
    """
    Save data to a CSV file.
    
    Args:
        data: DataFrame to save
        file_path: Path where to save the CSV file
    """
    try:
        data.to_csv(str(file_path), index=False)
    except Exception as e:
        logger.error(f"Error saving CSV file {file_path}: {str(e)}")
        raise

def normalize_data(data: np.ndarray,
                  scaler: StandardScaler = None) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize data using StandardScaler.
    
    Args:
        data: Data to normalize
        scaler: Optional pre-fitted StandardScaler
        
    Returns:
        Tuple of normalized data and fitted scaler
    """
    try:
        if scaler is None:
            scaler = StandardScaler()
            # Reshape data for scaling
            original_shape = data.shape
            data_reshaped = data.reshape(-1, original_shape[-1])
            # Fit and transform
            data_normalized = scaler.fit_transform(data_reshaped)
            # Reshape back
            return data_normalized.reshape(original_shape), scaler
        else:
            # Reshape data for scaling
            original_shape = data.shape
            data_reshaped = data.reshape(-1, original_shape[-1])
            # Transform
            data_normalized = scaler.transform(data_reshaped)
            # Reshape back
            return data_normalized.reshape(original_shape), scaler
    except Exception as e:
        logger.error(f"Error normalizing data: {str(e)}")
        raise

def create_data_loader(data: np.ndarray,
                         labels: np.ndarray,
                         batch_size: int = 32,
                         shuffle: bool = True) -> DataLoader:
    """
    Create a PyTorch data loader.
    
    Args:
        data: Input data
        labels: Target labels
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        
    Returns:
        PyTorch DataLoader
    """
    try:
        # Convert numpy arrays to torch tensors
        tensor_x = torch.Tensor(data)
        tensor_y = torch.Tensor(labels)
        
        dataset = TensorDataset(tensor_x, tensor_y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader
    except Exception as e:
        logger.error(f"Error creating data loader: {str(e)}")
        raise

def split_data(data: np.ndarray,
               labels: np.ndarray,
               train_ratio: float = 0.7,
               val_ratio: float = 0.15,
               test_ratio: float = 0.15,
               random_state: int = 42) -> Dict[str, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: Input data
        labels: Target labels
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        random_state: Random seed
        
    Returns:
        Dictionary containing split data
    """
    try:
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, \
            "Ratios must sum to 1"
        
        # Set random seed
        np.random.seed(random_state)
        
        # Get number of samples
        n_samples = len(data)
        
        # Generate random indices
        indices = np.random.permutation(n_samples)
        
        # Calculate split points
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        # Split indices
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Split data
        return {
            'train_data': data[train_indices],
            'train_labels': labels[train_indices],
            'val_data': data[val_indices],
            'val_labels': labels[val_indices],
            'test_data': data[test_indices],
            'test_labels': labels[test_indices]
        }
    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        raise

def load_adni_merge() -> pd.DataFrame:
    """
    Load the ADNI MERGE file.
    
    Returns:
        ADNI MERGE DataFrame
    """
    return load_csv(ADNI_MERGE_PATH)

def create_kfold_splits(data: np.ndarray, labels: np.ndarray, 
                       n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create k-fold cross-validation splits.
    
    Args:
        data: Input data
        labels: Target labels
        n_splits: Number of folds
        
    Returns:
        List of tuples with (train_data, train_labels, val_data, val_labels)
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []
    for train_index, val_index in kf.split(data):
        splits.append((data[train_index], labels[train_index], data[val_index], labels[val_index]))
    return splits

def get_class_distribution(labels: np.ndarray) -> Dict[int, int]:
    """
    Get the distribution of classes.
    
    Args:
        labels: Target labels
        
    Returns:
        Dictionary with class distribution
    """
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))

def load_nifti_image(file_path: Union[str, Path]) -> np.ndarray:
    return load_nifti(file_path)

def preprocess_image(image: np.ndarray,
                     target_size: Tuple[int, int, int] = IMAGE_SIZE,
                     normalize: bool = True,
                     resize: bool = True) -> np.ndarray:
    """
    Preprocess a single image.
    
    Args:
        image: Input image
        target_size: Target size for resizing
        normalize: Whether to normalize the image
        resize: Whether to resize the image
        
    Returns:
        Preprocessed image
    """
    from scipy.ndimage import zoom
    
    if resize:
        zoom_factors = [t / s for t, s in zip(target_size, image.shape)]
        image = zoom(image, zoom_factors, order=1)
        
    if normalize:
        image = (image - np.mean(image)) / np.std(image)
        
    return image 