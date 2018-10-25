#!/usr/bin/env python
# coding: utf-8

from utils import *


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
        rejects = files['rejects']
    except:
        pass

    # Clean data
    index, scaling_time, scalings = ['epoch', 'time'], 1e3, dict(grad=1e13)
    df = epochs.to_data_frame(picks=None, scalings=scalings, scaling_time=scaling_time, index=index)
    df_epochs = df.groupby('epoch').mean()
    X, y = df_epochs.values, rejects

    # Run algorithm
    y_pred = run_SVC(X, y)

    # Create .txt file with value 0 or 1 (reject) for each epoch
    np.savetxt('y_pred.txt', y_pred, fmt='%d', delimiter=',')


if __name__ == "__main__":
    main()
