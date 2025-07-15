# Running Guide for Alzheimer's Disease Analysis Pipeline (PyTorch Version)

This guide explains how to run the codebase, the purpose of each component, and the correct execution order.

## Prerequisites

1.  **Data Requirements**:
    *   ADNI dataset (ADNIMERGE.csv)
    *   3D MRI images in NIfTI format
    *   Place these in the `data/raw` directory

2.  **Environment Setup**:
    ```bash
    # Create and activate a virtual environment (recommended)
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    
    # Install required packages
    pip install -r new_scripts_pytorch/requirements.txt
    ```

3.  **Directory Structure**:
    The code will automatically create the following structure:
    ```
    project_root/
    ├── new_scripts_pytorch/ # All the project scripts
    ├── data/
    │   ├── raw/           # Original ADNI data
    │   ├── processed/     # Preprocessed data
    │   └── pickle/        # Serialized data
    ├── saved_models/      # Trained models
    ├── tensorboard_logs/  # Training logs
    └── results/           # Analysis results
    ```

## Execution Order

The following commands should be run from the `project_root` directory.

### 1. Data Preprocessing

These scripts are framework-agnostic and prepare the data for training.

```bash
python new_scripts_pytorch/data_preprocessing/preprocess_adni.py
python new_scripts_pytorch/data_preprocessing/preprocess_3d.py
python new_scripts_pytorch/data_preprocessing/mci_conversion_split.py
python new_scripts_pytorch/data_preprocessing/create_kfold_datasets.py
```

### 2. Model Training

These scripts will train the models and perform hyperparameter tuning using Optuna.

#### a. AD/CN Classification Model
```bash
python new_scripts_pytorch/model_training/train_ad_cn_model.py
```

#### b. MCI Conversion Model
```bash
python new_scripts_pytorch/model_training/train_mci_conversion_model.py
```

### 3. Analysis and Evaluation

#### a. Model Evaluation
```bash
python new_scripts_pytorch/analysis/evaluate_models.py
```

#### b. Uncertainty Analysis
```bash
python new_scripts_pytorch/analysis/analyze_uncertainty.py
```

#### c. Time to Conversion Analysis
```bash
python new_scripts_pytorch/analysis/analyze_time_to_conversion.py
```

## Utility Files

### 1. Configuration (`new_scripts_pytorch/utils/config.py`)
- Contains all project parameters, defines paths, and sets random seeds.

### 2. Data Utilities (`new_scripts_pytorch/utils/data_utils.py`)
- PyTorch-based data loading (`DataLoader`) and preprocessing functions.

### 3. Model Utilities (`new_scripts_pytorch/utils/model_utils.py`)
- PyTorch model definitions, training loops, and evaluation functions.

### 4. Visualization Utilities (`new_scripts_pytorch/utils/visualization_utils.py`)
- Plotting functions for metrics, confusion matrices, and other visualizations.

## Monitoring Training

1.  **TensorBoard**: The training scripts automatically log to TensorBoard.
    ```bash
    tensorboard --logdir=tensorboard_logs
    ```
    You can use it to view training progress, monitor metrics, and visualize model architecture.

2.  **Checkpoints**:
    - The best performing models are automatically saved in the `saved_models` directory during training.

## Troubleshooting

1.  **Memory Issues**:
    - Adjust `BATCH_SIZE` in `new_scripts_pytorch/utils/config.py`.
    - If you have a CUDA-enabled GPU, ensure PyTorch is installed with CUDA support.

2.  **Data Issues**:
    - Verify that the raw data is in the correct format and location (`data/raw`).
    - Ensure you have run all the data preprocessing steps before training.

## Output Interpretation

- **Metrics**: Check the `results` directory for CSV files containing performance metrics.
- **Plots**: Visualizations such as ROC curves, confusion matrices, and uncertainty distributions are also saved in the `results` directory.
- **Logs**: Console output provides real-time information on the training and analysis progress.

## Notes

- All paths are relative to the project root.
- Configuration can be modified in `new_scripts_pytorch/utils/config.py`.
- Logs are saved for debugging.
- Results are automatically organized in the `results` directory. 