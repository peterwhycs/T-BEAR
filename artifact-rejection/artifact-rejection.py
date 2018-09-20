#!/usr/bin/env python
# coding: utf-8

def main():
    import warnings
    from pathlib import Path, PureWindowsPath

    import matplotlib.pyplot as plt
    import mne
    import numpy as np
    import pandas as pd
    import scipy.io
    from sklearn.ensemble import IsolationForest
    warnings.filterwarnings('ignore')

    # Set file paths:
    print("Loading file paths...")
    file_path = str(Path(r'eeg-data/601/Rew_601_rest_bb_epoch.set'))
    mat_reject = str(Path(r'eeg-data/601/Rew_601_rest_reject_rmm.mat'))
    mat_stage = str(Path(r'eeg-data/601/Rew_601_rest_stages.mat'))

    # Load epochs file:
    print("Loading files...")
    try:
        epochs = mne.io.read_epochs_eeglab(file_path)
    except:
        epochs = mne.io.read_raw_eeglab(file_path)

    # Load sleep stages & reject files:
    try:
        sleep_file = scipy.io.loadmat(mat_stage)
        sleep = sleep_file['stages'].flatten()
    except FileNotFoundError:
        pass
    finally:
        reject_file = scipy.io.loadmat(mat_reject)
        reject = reject_file['reject'].flatten()

    # Convert to and clean DataFrame:
    print("Data cleaning...")
    df = epochs.to_data_frame()
    columns, df = sorted(list(df.columns)), df.reset_index()

    try: 
        df = df.drop(['condition'], axis=1)
    except:
        pass

    cleaned_columns = ['time']
    if 'epoch' in list(df.columns):
        cleaned_columns += ['epoch']

    cleaned_columns += columns
    df = df[cleaned_columns]
    df_ = df.copy()

    # Select values from columns for IForest:
    print("Preparing data for IForest algorithm...")
    value_columns = list(df.columns)

    try:
        if 'time' in value_columns:
            value_columns.remove('time')
        if 'epoch' in value_columns:
            value_columns.remove('epoch')
    except:
        pass

    df_values = df_[value_columns]

    # Run IForest:
    print("Running IForest algorithm...")
    X = df_values
    clfIF = IsolationForest(random_state=42, contamination=0.00001, n_jobs=3)
    clfIF.fit(X)
    pred_train, pred_test = clfIF.predict(X), clfIF.predict(X)
    count_train, count_test = np.unique(ar=pred_train, return_counts=True), np.unique(ar=pred_test, return_counts=True)
    index_train, index_test = [i for i,x in enumerate(pred_train) if x == -1] , [i for i,x in enumerate(pred_test) if x == -1]
    df_IF = df_.loc[index_test]
    num_artifacts = count_train[1][0], count_test[1][0]
    total_pts = count_train[1][1], count_test[1][1]
    total_artifacts = np.count_nonzero(reject)
    accuracy_percent = num_artifacts / total_artifacts * 100

    print("IForest algorithm completed!")
    print(f"Performance: {accuracy_percent} %")
    print(f"{num_artifacts} artifacts detected out of {total_artifacts} artifacts total.")


if __name__ == "__main__":
    main()
