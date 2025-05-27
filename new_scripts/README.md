# FDG MCI to AD Analysis - Scripts

This directory contains the modularized version of the original Jupyter notebooks, organized into a more maintainable and reusable structure.

## Directory Structure

```
scripts/
├── data_preprocessing/     # Data preparation and preprocessing scripts
├── model_training/        # Model training and tuning scripts
├── analysis/             # Analysis and results generation scripts
└── utils/                # Utility functions and shared code
```

## Workflow

### 1. Data Preprocessing
- `preprocess_adni.py`: Extract and organize data from ADNI database
- `preprocess_3d.py`: Preprocess AD/CN dataset
- `mci_conversion_split.py`: Divide MCI into pMCI and sMCI
- `create_kfold_datasets.py`: Create 5-fold cross-validation datasets

### 2. Model Training
- `train_ad_cn_model.py`: Train and tune AD/CN classification model
- `train_mci_progression.py`: Train MCI progression prediction models

### 3. Analysis
- `monte_carlo_analysis.py`: Generate predictions and uncertainty estimates
- `calculate_metrics.py`: Evaluate model performance
- `generate_results.py`: Create final analysis and correlations
- `time_to_conversion.py`: Analyze conversion time predictions

### 4. Utils
- `data_utils.py`: Data loading and processing utilities
- `model_utils.py`: Model architecture and training utilities
- `visualization_utils.py`: Plotting and visualization functions
- `config.py`: Configuration and constants

## Usage

1. First, ensure all required data is in place:
   - ADNI database access
   - Required directory structure
   - Configuration files

2. Run the preprocessing pipeline:
   ```bash
   python scripts/data_preprocessing/preprocess_adni.py
   python scripts/data_preprocessing/preprocess_3d.py
   python scripts/data_preprocessing/mci_conversion_split.py
   python scripts/data_preprocessing/create_kfold_datasets.py
   ```

3. Train the models:
   ```bash
   python scripts/model_training/train_ad_cn_model.py
   python scripts/model_training/train_mci_progression.py
   ```

4. Run analysis:
   ```bash
   python scripts/analysis/monte_carlo_analysis.py
   python scripts/analysis/calculate_metrics.py
   python scripts/analysis/generate_results.py
   python scripts/analysis/time_to_conversion.py
   ```

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Nibabel
- scikit-image
- cc3d
- pandas

## Configuration

Configuration parameters are stored in `utils/config.py`. Modify these parameters to adjust:
- Data paths
- Model parameters
- Training settings
- Analysis parameters 