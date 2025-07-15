"""
Data utilities for the FDG MCI to AD analysis project.
Contains functions for loading, processing, and saving data.
"""

import os
import sys
import numpy as np
import nibabel as nib
import pandas as pd
import tensorflow as tf
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

def create_data_generator(data: np.ndarray,
                         labels: np.ndarray,
                         batch_size: int = 32,
                         shuffle: bool = True) -> tf.data.Dataset:
    """
    Create a TensorFlow data generator.
    
    Args:
        data: Input data
        labels: Target labels
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        
    Returns:
        TensorFlow dataset
    """
    try:
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(data))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    except Exception as e:
        logger.error(f"Error creating data generator: {str(e)}")
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
    Load the ADNIMERGE.csv file.
    
    Returns:
        pd.DataFrame: ADNI merge data
    """
    return pd.read_csv(ADNI_MERGE_PATH)

def create_kfold_splits(data: np.ndarray, labels: np.ndarray, 
                       n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Create k-fold splits for cross-validation.
    
    Args:
        data: Array of data
        labels: Array of labels
        n_splits: Number of folds
        
    Returns:
        List of tuples containing (train_data, train_labels, val_data, val_labels)
    """
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    splits = []
    for train_idx, val_idx in kf.split(data):
        train_data, train_labels = data[train_idx], labels[train_idx]
        val_data, val_labels = data[val_idx], labels[val_idx]
        splits.append((train_data, train_labels, val_data, val_labels))
    
    return splits

def get_class_distribution(labels: np.ndarray) -> Dict[int, int]:
    """
    Get the distribution of classes in the dataset.
    
    Args:
        labels: Array of labels
        
    Returns:
        Dict mapping class indices to counts
    """
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts)) 

# ----------------------------------------------------------------------------
# Compatibility helpers (used by older preprocessing scripts)
# ----------------------------------------------------------------------------

def load_nifti_image(file_path: Union[str, Path]) -> np.ndarray:
    """Convenience wrapper around :func:`load_nifti` for legacy code."""
    return load_nifti(file_path)


def preprocess_image(image: np.ndarray,
                     target_size: Tuple[int, int, int] = IMAGE_SIZE,
                     normalize: bool = True,
                     resize: bool = True) -> np.ndarray:
    """Resize and/or z-score-normalise a 3-D image volume.

    Parameters
    ----------
    image : np.ndarray
        Raw 3-D image data.
    target_size : tuple
        Desired output shape (x, y, z).
    normalize : bool
        Whether to z-score the voxels (mean-0, std-1).
    resize : bool
        Whether to resample the volume to *target_size* using trilinear
        interpolation.
    """
    if resize and image.shape != target_size:
        # Import lazily to keep the base import footprint small.
        from skimage.transform import resize as _resize
        image = _resize(image, target_size, order=1, mode="constant", anti_aliasing=True)

    if normalize:
        mean = image.mean()
        std = image.std() or 1.0
        image = (image - mean) / std

    return image.astype(np.float32)


def create_tf_dataset(data: np.ndarray,
                      labels: np.ndarray,
                      batch_size: int = BATCH_SIZE,
                      shuffle: bool = True) -> tf.data.Dataset:
    """Alias for :func:`create_data_generator` kept for backward compatibility."""
    return create_data_generator(data, labels, batch_size=batch_size, shuffle=shuffle) 