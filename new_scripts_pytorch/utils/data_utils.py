"""
PyTorch-friendly data utilities mirroring the public API of
``new_scripts.utils.data_utils`` but implemented without any TensorFlow
dependency. Only the functions required by the ported training / evaluation
code are included for now.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import logging

import numpy as np
import nibabel as nib
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, Dataset

from .config import IMAGE_SIZE, BATCH_SIZE, RANDOM_STATE

# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Basic IO helpers (nifti / pickle / csv)
# ----------------------------------------------------------------------------
import pickle
import pandas as pd

def load_nifti(file_path: Union[str, Path]) -> np.ndarray:
    """Load a NIfTI file and return its data as a numpy array."""
    file_path = Path(file_path)
    try:
        nifti_img = nib.load(str(file_path))
        return nifti_img.get_fdata()
    except Exception as e:
        logger.error("Error loading NIfTI file %s: %s", file_path, e)
        raise

def save_nifti(data: np.ndarray, file_path: Union[str, Path], affine: np.ndarray | None = None) -> None:
    """Save *data* to *file_path* as a NIfTI volume."""
    file_path = Path(file_path)
    try:
        if affine is None:
            affine = np.eye(4)
        nifti_img = nib.Nifti1Image(data, affine)
        nib.save(nifti_img, str(file_path))
    except Exception as e:
        logger.error("Error saving NIfTI file %s: %s", file_path, e)
        raise

def load_pickle(file_path: Union[str, Path]) -> Any:
    file_path = Path(file_path)
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error("Error loading pickle file %s: %s", file_path, e)
        raise

def save_pickle(data: Any, file_path: Union[str, Path]) -> None:
    file_path = Path(file_path)
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.error("Error saving pickle file %s: %s", file_path, e)
        raise

def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error("Error loading CSV %s: %s", file_path, e)
        raise

def save_csv(df: pd.DataFrame, file_path: Union[str, Path]) -> None:
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        logger.error("Error saving CSV %s: %s", file_path, e)
        raise

# ----------------------------------------------------------------------------
# Normalisation & preprocessing
# ----------------------------------------------------------------------------

def normalize_data(data: np.ndarray, scaler: StandardScaler | None = None) -> Tuple[np.ndarray, StandardScaler]:
    """Normalise *data* using a ``StandardScaler`` (feature-wise)."""
    try:
        if scaler is None:
            scaler = StandardScaler()
            reshaped = data.reshape(-1, data.shape[-1])
            data_norm = scaler.fit_transform(reshaped).reshape(data.shape)
            return data_norm, scaler
        else:
            reshaped = data.reshape(-1, data.shape[-1])
            data_norm = scaler.transform(reshaped).reshape(data.shape)
            return data_norm, scaler
    except Exception as e:
        logger.error("Error normalising data: %s", e)
        raise

# ----------------------------------------------------------------------------
# PyTorch dataset helpers
# ----------------------------------------------------------------------------

def create_data_loader(data: np.ndarray, labels: np.ndarray, batch_size: int = BATCH_SIZE, shuffle: bool = True) -> DataLoader:
    """Create a ``torch.utils.data.DataLoader`` from in-memory ``numpy`` arrays."""
    try:
        data_tensor = torch.from_numpy(data).float()
        labels_tensor = torch.from_numpy(labels).long()
        dataset = TensorDataset(data_tensor, labels_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader
    except Exception as e:
        logger.error("Error creating DataLoader: %s", e)
        raise

# ----------------------------------------------------------------------------
# Data splitting utilities (similar to original TensorFlow version)
# ----------------------------------------------------------------------------

def split_data(
    data: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> Dict[str, np.ndarray]:
    """Split dataset into train/val/test numpy arrays."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1."
    np.random.seed(random_state)
    n_samples = len(data)
    indices = np.random.permutation(n_samples)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]
    return {
        'train_data': data[train_idx],
        'train_labels': labels[train_idx],
        'val_data': data[val_idx],
        'val_labels': labels[val_idx],
        'test_data': data[test_idx],
        'test_labels': labels[test_idx],
    }

# ----------------------------------------------------------------------------
# K-fold splitting (returns numpy arrays, caller decides on DataLoader)
# ----------------------------------------------------------------------------
from sklearn.model_selection import KFold

def create_kfold_splits(
    data: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return list with tuples ``(train_x, train_y, val_x, val_y)`` for each fold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for train_idx, val_idx in kf.split(data):
        splits.append((data[train_idx], labels[train_idx], data[val_idx], labels[val_idx]))
    return splits 