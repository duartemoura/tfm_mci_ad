"""
Configuration settings for the project.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
PICKLE_DIR = DATA_DIR / 'pickle'

# Model directories
SAVED_MODELS_DIR = PROJECT_ROOT / 'saved_models'
TENSORBOARD_DIR = PROJECT_ROOT / 'tensorboard_logs'
KERAS_TUNER_DIR = PROJECT_ROOT / 'keras_tuner'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Create directories if they don't exist
for directory in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, PICKLE_DIR,
    SAVED_MODELS_DIR, TENSORBOARD_DIR, KERAS_TUNER_DIR, RESULTS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# Data parameters
IMAGE_SIZE = (96, 96, 96)  # Standard size for 3D images
BATCH_SIZE = 32
NUM_CLASSES = 2  # Binary classification
N_SPLITS = 5  # Number of folds for cross-validation

# Model parameters
LEARNING_RATE = 1e-4
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# ADNI data paths
ADNI_MERGE_PATH = RAW_DATA_DIR / 'ADNIMERGE.csv'
ADNI_IMAGE_DIR = RAW_DATA_DIR / 'images'

# Preprocessing parameters
NORMALIZE = True
RESIZE = True
AUGMENT = True

# Augmentation parameters
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
ZOOM_RANGE = 0.1
HORIZONTAL_FLIP = True
FILL_MODE = 'nearest'

# Training parameters
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_STATE = 42

# Logging parameters
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# GPU settings
GPU_MEMORY_GROWTH = True
GPU_MEMORY_LIMIT = 4096  # MB

# Set GPU memory growth
if GPU_MEMORY_GROWTH:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if GPU_MEMORY_LIMIT:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(
                        memory_limit=GPU_MEMORY_LIMIT
                    )]
                )
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")

# Set random seeds for reproducibility
import numpy as np
import tensorflow as tf
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Model checkpoints
BEST_AD_CN_MODEL = "3d_g_tuned/epoch29"  # Best performing model for AD/CN classification 