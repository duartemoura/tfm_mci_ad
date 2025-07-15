"""
Configuration settings for the PyTorch re-implementation of the project.
This file mirrors the structure of the original ``new_scripts.utils.config`` module
so that ported code can keep the same import paths / constant names while relying
on PyTorch instead of TensorFlow.
"""

from pathlib import Path
import os
import torch
import numpy as np

# -----------------------------------------------------------------------------
# Project & directory layout
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
PICKLE_DIR = DATA_DIR / 'pickle'

# Model directories
SAVED_MODELS_DIR = PROJECT_ROOT / 'saved_models_pytorch'
TENSORBOARD_DIR = PROJECT_ROOT / 'tensorboard_logs_pytorch'
KERAS_TUNER_DIR = PROJECT_ROOT / 'optuna'               # kept for backwards-compat
RESULTS_DIR = PROJECT_ROOT / 'results_pytorch'

# Create directories if they don't exist
for directory in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, PICKLE_DIR,
    SAVED_MODELS_DIR, TENSORBOARD_DIR, KERAS_TUNER_DIR, RESULTS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Data / model hyper-parameters (mirrors original defaults)
# -----------------------------------------------------------------------------
IMAGE_SIZE = (96, 96, 96)   # depth, height, width for 3D volumes
BATCH_SIZE = 32
NUM_CLASSES = 2
N_SPLITS = 5

# Backwards-compatibility aliases ------------------------------------------------
NEW_DATA_DIR = PROCESSED_DATA_DIR / 'images'
NUM_FOLDS = N_SPLITS
# -----------------------------------------------------------------------------

# Optimiser / training parameters
LEARNING_RATE = 1e-4
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10

# Randomness --------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# GPU settings ------------------------------------------------------------------
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# Optionally, limit GPU memory (if desired). PyTorch does not offer an exact
# analogue of TensorFlow's logical device config, but we can set the following
# environment variable to ask CUDA to reserve less memory up-front.
GPU_MEMORY_LIMIT_MB = 4096  # set to ``None`` to disable
if GPU_MEMORY_LIMIT_MB is not None and USE_CUDA:
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', f'max_split_size_mb:{GPU_MEMORY_LIMIT_MB}')

# Model checkpoints
BEST_AD_CN_MODEL = SAVED_MODELS_DIR / 'best_ad_cn_model.pt' 