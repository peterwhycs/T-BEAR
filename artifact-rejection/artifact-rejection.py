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

    epoch_3d_array = epochs.get_data()

    X, y = epoch_3d_array, rejects
    y_pred = run_SVM(epoch_3d_array, rejects):

    # Create .csv file with rejected <#> epochs
    y_pred.tofile('y_pred.csv', sep=',', format='%0f')


if __name__ == "__main__":
    main()
