"""
Configuration settings for the project.
"""

import os
from pathlib import Path
import torch
import numpy as np

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
RESULTS_DIR = PROJECT_ROOT / 'results'

# Create directories if they don't exist
for directory in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, PICKLE_DIR,
    SAVED_MODELS_DIR, TENSORBOARD_DIR, RESULTS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# Data parameters
IMAGE_SIZE = (96, 96, 96)  # Standard size for 3D images
BATCH_SIZE = 32
NUM_CLASSES = 2  # Binary classification
N_SPLITS = 5  # Number of folds for cross-validation

# Backwards-compatibility constants -------------------------------------------------
# Some legacy modules still expect the names ``NEW_DATA_DIR`` and ``NUM_FOLDS``.
# We map those to the new canonical locations so that both old and new code work.
NEW_DATA_DIR = PROCESSED_DATA_DIR / 'images'
NUM_FOLDS = N_SPLITS
# -----------------------------------------------------------------------------------

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

# Set random seeds for reproducibility
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# Model checkpoints
BEST_AD_CN_MODEL = "3d_g_tuned/epoch29"  # Best performing model for AD/CN classification 