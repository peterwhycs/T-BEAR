#!/usr/bin/env python
# coding: utf-8

from utils import *


def main():
    # Load researcher and subject folders
    subjects = list()
    stephanie_folder = Path("C:\\Users\\peter\\git\\EEG-artifact-rejection\\artifact-rejection\\eeg-data\\Stephanie")
    sub_folders = [os.path.join(stephanie_folder, file) for file in os.listdir(stephanie_folder)]

    # Use the median for each of the bad channels
    stephanie_bad_chan = {
        'Rew_605_rest': ['Fp1', 'Fp2'],
        'Rew_609_rest': ['F3', 'F7', 'Fp1'],
        'Rew_611_rest': ['Fp2', 'T3'],
        'Rew_613_rest': ['F7', 'Fp2'],
        'Rew_614_rest': ['C3', 'CZ', 'F3', 'F7', 'F8', 'FZ', 'Fp1', 'O1', 'P3', 'PZ', 'T3', 'T5'],
        'Rew_615_rest': ['C4', 'F4', 'F8', 'Fp2', 'O2', 'P4', 'T4', 'T6'],
        'Rew_619_rest': ['F4'],
        'Rew_622_rest': ['F7'],
        'Rew_624_rest': ['F3', 'F7', 'Fp1', 'T3', 'T4', 'T5', 'T6'],
        'Rew_626_rest': ['F3', 'F4', 'T3', 'T4', 'T5'],
        'Rew_701_rest': ['C4', 'F7', 'F8', 'O1', 'O2', 'T3', 'T3', 'T4', 'T5'],
        'Rew_702_rest': ['C3', 'F3'],
        'Rew_703_rest': ['F4', 'F7', 'F8', 'Fp2', 'T3', 'T4', 'T6'],
        'Rew_704_rest': ['C3'],
        'Rew_706_rest': ['T4']
    }

    # Add file name and paths to dictionary
    for sub in sub_folders:
        files = os.listdir(Path(sub))
        temp_sub_files = dict()
        for file in files:
            file_path = os.path.join(Path(sub), file)
            temp_sub_files['id'] = str(sub)
            if 'epoch' in file:
                temp_sub_files['epoch'] = file_path
            if 'reject' in file:
                temp_sub_files['reject'] = file_path
            elif 'stages' in file:
                temp_sub_files['stage'] = file_path
        subjects.append(temp_sub_files)

    # Initialize classifier
    clfSVC = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=0.0001, C=10.0, multi_class='ovr', fit_intercept=True,
                       intercept_scaling=1, class_weight=None, verbose=1, random_state=42, max_iter=1000)

    # Model training
    for sub_ in x_train:
        file_path = sub_['epoch']
        mat_reject = sub_['reject']
        mat_stage = sub_['stage']

        files = load_subject_dir(file_path, mat_reject, mat_stage)
        epochs = files['epochs']
        rejects = files['rejects']

        # Clean data
        index, scaling_time, scalings = ['epoch', 'time'], 1e3, dict(grad=1e13)
        df = epochs.to_data_frame(
            picks=None, scalings=scalings, scaling_time=scaling_time, index=index)
        df_epochs = df.groupby('epoch').mean()

        try:
            stages = files['stages']
            df_epochs['stage'] = stages
        except Exception as ex:
            print(ex)
            pass

        df_epochs = df.groupby('epoch').mean()
        tscv = TimeSeriesSplit(n_splits=3)
        for train_index, test_index in tscv.split(df_epochs):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

        X, y = df_epochs.values, rejects
        X, y_true = X, y
        clfSVC.fit(X, y_true)

    # Model testing
    for sub__ in x_test:
        file_path = sub__['epoch']
        mat_reject = sub__['reject']
        mat_stage = sub__['stage']

        files = load_subject_dir(file_path, mat_reject, mat_stage)
        epochs = files['epochs']
        rejects = files['rejects']

        # Clean data
        index, scaling_time, scalings = ['epoch', 'time'], 1e3, dict(grad=1e13)
        df = epochs.to_data_frame(
            picks=None, scalings=scalings, scaling_time=scaling_time, index=index)
        df_epochs = df.groupby('epoch').mean()

        try:
            stages = files['stages']
            df_epochs['stage'] = stages
        except Exception as ex:
            print(ex)
            pass

        df_epochs = df.groupby('epoch').mean()
        X, y = df_epochs.values, rejects
        X, y_true = X, y
        y_pred = clfSVC.predict(X)

        print("\tRecall: %1.3f" % recall_score(y_true, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_true, y_pred))


if __name__ == "__main__":
    main()
