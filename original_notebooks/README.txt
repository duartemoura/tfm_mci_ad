Instructions for the project.

Folders:
-datasets: TF dataset objetcs used.
-excels: data from the project.
-keras_tuner: KT data.
-my_log_dir: tensorboard data.
-old_notebooks.
-pdata: imaging data. ONLY newdata is useful, rest is from previous.
-pickle: loading pickles for datasets.
-saved_models: checkpoints saved.

Notebooks:
-3D_preprocess: preprocess for AD/CN dataset.
-3D tuning and training: for AD/CN.
-Demographics: get from ADNIMERGE.csv all variables.
-kfold for MCI prog: creates 5 folds and preprocessed TF objects with them.
-MCI_conversion_split: divides MCI into pMCI and sMCI.
-Metrics: gets metrics from model performance.
-MMSE and MCI(DEPRECATED): gets MMSE data from PET metadata. N was low, we used MMSE data from Demographics.
-MonteCarlo: gets MCD predictions, SD and REGULAR PREDICTIONS for each patient. This goes into the Results notebook and final results. If you NEED THE FINAL pMCI/sMCI MODEL, THE NAME WILL BE HERE IN THE MONTECARLO.
-Prepreprocess: Gets data from ADNI folder into pdata/newdata.

Workflow for the project:
-Prepreprocess: get data from ADNI and divide into AD/CN/MCI
-3D_preprocess: preprocess AD/CN dataset.
-3D_tuning_train: KT for AD/CN and then train. BEST PERFORMING MODEL AND THE ONE WE USED FOR MOVING ON TO MCI(after correct replication) WAS: 3d_g_tuned/epoch29.
-MCI_conv_split: divide MCI according to conversion.
-kfold: get MCI data in 5 folds and 5 datasets.
-MCI to AD train: train 5 models, one per fold.
-Demographics/Metrics/MMSE/MonteCarlo/TTC: all get additional data from trained model and other variables.
-Results: pool all the information from predictions and other variables and performs correlations, table1, etc.
