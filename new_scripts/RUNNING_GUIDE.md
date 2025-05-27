# Running Guide for Alzheimer's Disease Analysis Pipeline

This guide explains how to run the codebase, the purpose of each component, and the correct execution order.

## Prerequisites

1. **Data Requirements**:
   - ADNI dataset (ADNIMERGE.csv)
   - 3D MRI images in NIfTI format
   - Place these in the `data/raw` directory

2. **Environment Setup**:
   ```bash
   # Create and activate a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install required packages
   pip install -r requirements.txt
   ```

3. **Directory Structure**:
   The code will automatically create the following structure:
   ```
   project_root/
   ├── data/
   │   ├── raw/           # Original ADNI data
   │   ├── processed/     # Preprocessed data
   │   └── pickle/        # Serialized data
   ├── saved_models/      # Trained models
   ├── tensorboard_logs/  # Training logs
   ├── keras_tuner/       # Hyperparameter tuning results
   └── results/           # Analysis results
   ```

## Execution Order

### 1. Data Preprocessing

#### a. ADNI Data Preprocessing
```bash
python data_preprocessing/preprocess_adni.py
```
- **Input**: ADNIMERGE.csv in `data/raw`
- **Output**: Processed ADNI data in `data/processed`
- **Purpose**: Cleans and prepares ADNI demographic and clinical data

#### b. 3D Image Preprocessing
```bash
python data_preprocessing/preprocess_3d.py
```
- **Input**: Raw 3D MRI images in `data/raw/images`
- **Output**: Preprocessed images in `data/processed/images`
- **Purpose**: Normalizes and resizes 3D MRI images

#### c. MCI Conversion Split
```bash
python data_preprocessing/mci_conversion_split.py
```
- **Input**: Processed ADNI data
- **Output**: MCI conversion dataset in `data/processed`
- **Purpose**: Creates dataset for MCI conversion prediction

#### d. Create K-Fold Datasets
```bash
python data_preprocessing/create_kfold_datasets.py
```
- **Input**: Preprocessed data
- **Output**: K-fold datasets in `data/processed`
- **Purpose**: Creates cross-validation splits

### 2. Model Training

#### a. AD/CN Classification Model
```bash
python model_training/train_ad_cn_model.py
```
- **Input**: Preprocessed AD/CN data
- **Output**: Trained model in `saved_models`
- **Purpose**: Trains model to classify AD vs CN

#### b. MCI Conversion Model
```bash
python model_training/train_mci_conversion_model.py
```
- **Input**: Preprocessed MCI conversion data
- **Output**: Trained model in `saved_models`
- **Purpose**: Trains model to predict MCI conversion

### 3. Analysis and Evaluation

#### a. Model Evaluation
```bash
python analysis/evaluate_models.py
```
- **Input**: Trained models and test data
- **Output**: Performance metrics in `results`
- **Purpose**: Evaluates model performance

#### b. Time to Conversion Analysis
```bash
python analysis/analyze_time_to_conversion.py
```
- **Input**: MCI conversion predictions
- **Output**: Time analysis results in `results`
- **Purpose**: Analyzes conversion timing

#### c. Uncertainty Analysis
```bash
python analysis/analyze_uncertainty.py
```
- **Input**: Model predictions
- **Output**: Uncertainty metrics in `results`
- **Purpose**: Analyzes prediction uncertainty

## Utility Files

### 1. Configuration (`utils/config.py`)
- Contains all project parameters
- Defines paths and hyperparameters
- Manages GPU settings

### 2. Data Utilities (`utils/data_utils.py`)
- Data loading and preprocessing functions
- Data augmentation
- Batch generation

### 3. Model Utilities (`utils/model_utils.py`)
- Model architecture definitions
- Training functions
- Model evaluation functions

### 4. Visualization Utilities (`utils/visualization_utils.py`)
- Plotting functions
- Results visualization
- Performance metrics visualization

## Monitoring Training

1. **TensorBoard**:
   ```bash
   tensorboard --logdir=tensorboard_logs
   ```
   - View training progress
   - Monitor metrics
   - Visualize model architecture

2. **Checkpoints**:
   - Models are saved in `saved_models`
   - Best models are automatically saved
   - Training can be resumed from checkpoints

## Troubleshooting

1. **Memory Issues**:
   - Adjust batch size in `config.py`
   - Modify GPU memory limit
   - Use data generators for large datasets

2. **Data Issues**:
   - Verify data format
   - Check file permissions
   - Ensure correct directory structure

3. **Training Issues**:
   - Monitor GPU usage
   - Check learning rate
   - Verify data preprocessing

## Output Interpretation

1. **Model Performance**:
   - Check `results` directory for metrics
   - Review confusion matrices
   - Analyze ROC curves

2. **Conversion Analysis**:
   - Review time-to-conversion plots
   - Check uncertainty metrics
   - Analyze feature importance

## Notes

- All paths are relative to the project root
- Configuration can be modified in `config.py`
- Logs are saved for debugging
- Results are automatically organized in the `results` directory 