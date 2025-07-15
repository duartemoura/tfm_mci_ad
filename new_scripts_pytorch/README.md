# FDG MCI to AD Analysis - PyTorch Scripts

This directory contains a PyTorch version of the original project, migrated from TensorFlow. The scripts are organized into a modular structure for maintainability and reusability.

## Directory Structure

```
new_scripts_pytorch/
├── data_preprocessing/     # Data preparation and preprocessing scripts
├── model_training/        # Model training and tuning scripts
├── analysis/             # Analysis and results generation scripts
└── utils/                # Utility functions and shared code
```

## Workflow

### 1. Data Preprocessing
The data preprocessing scripts are framework-agnostic and remain the same as the original version.
- `preprocess_adni.py`: Extract and organize data from ADNI database
- `preprocess_3d.py`: Preprocess AD/CN dataset
- `mci_conversion_split.py`: Divide MCI into pMCI and sMCI
- `create_kfold_datasets.py`: Create 5-fold cross-validation datasets

### 2. Model Training
- `train_ad_cn_model.py`: Train and tune AD/CN classification model using PyTorch and Optuna.
- `train_mci_conversion_model.py`: Train and tune MCI conversion prediction model with k-fold cross-validation using PyTorch and Optuna.

### 3. Analysis
- `evaluate_models.py`: Evaluate model performance for both AD/CN and MCI conversion models.
- `analyze_uncertainty.py`: Generate predictions and uncertainty estimates using Monte Carlo dropout.
- `analyze_time_to_conversion.py`: Analyze time to conversion and MMSE scores for MCI patients.

### 4. Utils
- `data_utils.py`: PyTorch-based data loading and processing utilities.
- `model_utils.py`: PyTorch model architecture, training, and evaluation utilities.
- `visualization_utils.py`: Plotting and visualization functions.
- `config.py`: Configuration and constants for the PyTorch project.

## Usage

1.  **Setup Environment**: Install the required packages from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Data Preprocessing**: Execute the preprocessing scripts to prepare the datasets.
    ```bash
    python new_scripts_pytorch/data_preprocessing/preprocess_adni.py
    python new_scripts_pytorch/data_preprocessing/preprocess_3d.py
    # ... and other preprocessing steps
    ```

3.  **Train Models**: Run the training scripts. This will also perform hyperparameter tuning with Optuna.
    ```bash
    python new_scripts_pytorch/model_training/train_ad_cn_model.py
    python new_scripts_pytorch/model_training/train_mci_conversion_model.py
    ```

4.  **Run Analysis**: Execute the analysis scripts to evaluate the models and generate results.
    ```bash
    python new_scripts_pytorch/analysis/evaluate_models.py
    python new_scripts_pytorch/analysis/analyze_uncertainty.py
    python new_scripts_pytorch/analysis/analyze_time_to_conversion.py
    ```

## Requirements

- Python 3.x
- PyTorch
- Optuna
- NumPy
- Matplotlib
- Nibabel
- scikit-learn
- scikit-image
- pandas
- seaborn
- SimpleITK

## Configuration

Configuration parameters are stored in `utils/config.py`. Modify these parameters to adjust:
- Data paths
- Model parameters (hyperparameters are tuned automatically)
- Training settings
- Analysis parameters 