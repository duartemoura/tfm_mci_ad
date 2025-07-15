"""
Create k-fold cross-validation datasets for MCI progression prediction.
This script creates 5-fold cross-validation datasets from the pMCI/sMCI data.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, List, Dict
import tensorflow as tf

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.data_utils import (
    create_tf_dataset, save_pickle, create_kfold_splits
)
from scripts.utils.config import (
    NEW_DATA_DIR, PICKLE_DIR, NUM_FOLDS, BATCH_SIZE
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_mci_images(group: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images for a specific MCI group (pMCI or sMCI).
    
    Args:
        group: Group name ('pMCI' or 'sMCI')
        
    Returns:
        Tuple of (images, labels)
    """
    group_dir = NEW_DATA_DIR / group
    if not group_dir.exists():
        raise ValueError(f"Directory {group_dir} does not exist")
    
    images = []
    labels = []
    
    # Load all .npy files in the group directory
    for file_path in group_dir.glob('*.npy'):
        try:
            image = np.load(file_path)
            images.append(image)
            labels.append(1 if group == 'pMCI' else 0)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            continue
    
    return np.array(images), np.array(labels)

def create_kfold_datasets() -> List[Dict[str, tf.data.Dataset]]:
    """
    Create k-fold cross-validation datasets.
    
    Returns:
        List of dictionaries containing train and validation datasets for each fold
    """
    # Load pMCI and sMCI images
    pmci_images, pmci_labels = load_mci_images('pMCI')
    logger.info(f"Loaded {len(pmci_images)} pMCI images")
    
    smci_images, smci_labels = load_mci_images('sMCI')
    logger.info(f"Loaded {len(smci_images)} sMCI images")
    
    # Combine data
    all_images = np.concatenate([pmci_images, smci_images])
    all_labels = np.concatenate([pmci_labels, smci_labels])
    
    # Create k-fold splits
    splits = create_kfold_splits(all_images, all_labels, n_splits=NUM_FOLDS)
    
    # Create TensorFlow datasets for each fold
    fold_datasets = []
    for i, (train_images, train_labels, val_images, val_labels) in enumerate(splits):
        train_dataset = create_tf_dataset(train_images, train_labels)
        val_dataset = create_tf_dataset(val_images, val_labels)
        
        fold_datasets.append({
            'fold': i + 1,
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'train_images': train_images,
            'train_labels': train_labels,
            'val_images': val_images,
            'val_labels': val_labels
        })
        
        logger.info(f"Created fold {i + 1} with {len(train_images)} training and {len(val_images)} validation samples")
    
    # Save fold datasets
    save_pickle(fold_datasets, 'mci_kfold_datasets.pkl')
    
    return fold_datasets

def main():
    """Main function to run the k-fold dataset creation pipeline."""
    logger.info("Starting k-fold dataset creation...")
    
    try:
        fold_datasets = create_kfold_datasets()
        logger.info(f"Successfully created {len(fold_datasets)} fold datasets")
        
    except Exception as e:
        logger.error(f"Error in k-fold dataset creation pipeline: {str(e)}")
        sys.exit(1)
    
    logger.info("K-fold dataset creation completed")

if __name__ == "__main__":
    main() 