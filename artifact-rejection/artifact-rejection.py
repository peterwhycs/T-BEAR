#!/usr/bin/env python
# coding: utf-8

from utilities import *


def main():
    # Set file paths and load files
    file_path = Path(
        '/home/walker/peterwhy/git/EEG-artifact-rejection/artifact-rejection/eeg-data/Stephanie/Rew_601_rest/Rew_601_rest_bb_epoch.set')
    mat_reject = Path(
        '/home/walker/peterwhy/git/EEG-artifact-rejection/artifact-rejection/eeg-data/Stephanie/Rew_601_rest/Rew_601_rest_reject_rmm.mat')
    mat_stage = Path(
        '/home/walker/peterwhy/git/EEG-artifact-rejection/artifact-rejection/eeg-data/Stephanie/Rew_601_rest/Rew_601_rest_stages.mat')

    files = load_subject_dir(file_path, mat_reject, mat_stage)
    epochs = files['epochs']
    try:
        reject = files['reject']
    except:
        pass

    # Read epochs into dataframe and clean df
    index, scaling_time, scalings = ['epoch', 'time'], 1e3, dict(grad=1e13)
    df_read = epochs.to_data_frame(picks=None, scalings=scalings, scaling_time=scaling_time, index=index)
    df = clean_df(df_read)
    df_ = df.copy()

    # Select values from dataframe and reject file
    df_values = extract_df_values(df)
    X, y = df_values.values, reject
    df_IF = run_IForest(X, y, df)

    # Create reject array .csv file
    y_pred_reject = reject_epochs(df, df_IF)
    y_pred_= np.asarray(y_pred_reject)
    y_pred_.tofile('y_pred.csv',sep=',',format='%1f')
    acc_score = accuracy_score(reject, y_pred_)
    print(acc_score)


if __name__ == "__main__":
    main()
