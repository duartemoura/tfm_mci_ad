"""
Preprocess 3D images for AD/CN classification.
This script handles the preprocessing of the AD/CN dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, List, Dict

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from scripts.utils.data_utils import (
    load_nifti_image, preprocess_image, save_pickle,
    create_tf_dataset
)
from scripts.utils.config import (
    NEW_DATA_DIR, PICKLE_DIR, IMAGE_SIZE, BATCH_SIZE
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess_images(class_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess all images for a given class.
    
    Args:
        class_name: Name of the class ('AD' or 'CN')
        
    Returns:
        Tuple of (images, labels)
    """
    class_dir = NEW_DATA_DIR / class_name
    if not class_dir.exists():
        raise ValueError(f"Directory {class_dir} does not exist")
    
    images = []
    labels = []
    
    # Load all .npy files in the class directory
    for file_path in class_dir.glob('*.npy'):
        try:
            image = np.load(file_path)
            images.append(image)
            labels.append(1 if class_name == 'AD' else 0)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            continue
    
    return np.array(images), np.array(labels)

def create_ad_cn_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create TensorFlow datasets for AD/CN classification.
    
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Load and preprocess AD images
    ad_images, ad_labels = load_and_preprocess_images('AD')
    logger.info(f"Loaded {len(ad_images)} AD images")
    
    # Load and preprocess CN images
    cn_images, cn_labels = load_and_preprocess_images('CN')
    logger.info(f"Loaded {len(cn_images)} CN images")
    
    # Combine data
    all_images = np.concatenate([ad_images, cn_images])
    all_labels = np.concatenate([ad_labels, cn_labels])
    
    # Shuffle data
    indices = np.random.permutation(len(all_images))
    all_images = all_images[indices]
    all_labels = all_labels[indices]
    
    # Split into train and validation sets (80/20)
    split_idx = int(0.8 * len(all_images))
    train_images = all_images[:split_idx]
    train_labels = all_labels[:split_idx]
    val_images = all_images[split_idx:]
    val_labels = all_labels[split_idx:]
    
    # Create TensorFlow datasets
    train_dataset = create_tf_dataset(train_images, train_labels)
    val_dataset = create_tf_dataset(val_images, val_labels)
    
    # Save preprocessed data
    save_pickle({
        'train_images': train_images,
        'train_labels': train_labels,
        'val_images': val_images,
        'val_labels': val_labels
    }, 'ad_cn_preprocessed.pkl')
    
    return train_dataset, val_dataset

def main():
    """Main function to run the 3D preprocessing pipeline."""
    logger.info("Starting 3D preprocessing for AD/CN dataset...")
    
    try:
        train_dataset, val_dataset = create_ad_cn_dataset()
        logger.info("Successfully created AD/CN datasets")
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        sys.exit(1)
    
    logger.info("3D preprocessing completed")

if __name__ == "__main__":
    main() 