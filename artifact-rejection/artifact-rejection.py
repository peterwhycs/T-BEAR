#!/usr/bin/env python
# coding: utf-8

from utils import *


def main():
    # Use the median for each of the bad channels
    bad_channels_stephanie = {
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

    # Load researcher and subject folders
    stephanie_folder = Path("C:\\Users\\peter\\git\\EEG-artifact-rejection\\artifact-rejection\\eeg-data\\Stephanie")
    sub_folders = [[file, os.path.join(stephanie_folder, file)] for file in os.listdir(stephanie_folder)]
    train_sub_folders , test_sub_folders = train_test_split(sub_folders)

    subjects = dict()
    for sub in train_sub_folders:
        sub_id, path_ = sub[0], sub[1]
        files = os.listdir(Path(path_))
        sub_files = dict()
        for file in files:
            full_path = os.path.join(Path(path_), file)
            if 'epoch' in file:
                sub_files['epoch'] = full_path
            if 'reject' in file:
                sub_files['reject'] = full_path
            elif 'stages' in file:
                sub_files['stage'] = full_path
        subjects[sub_id] = sub_files

    epoched_dataframes = list()
    for sub_ in subjects.keys():
        file_path = subjects[sub_]['epoch']
        mat_reject = subjects[sub_]['reject']
        mat_stage = subjects[sub_]['stage']

        files = load_subject_dir(file_path, mat_reject, mat_stage)
        epochs = files['epochs']
        rejects = files['rejects']

        index, scaling_time, scalings = ['epoch', 'time'], 1e3, dict(grad=1e13)
        df = epochs.to_data_frame(
            picks=None, scalings=scalings, scaling_time=scaling_time, index=index)
        df_epochs = df.groupby('epoch').mean()

        try:
            stages = files['stages']
            df_epochs['stage'] = stages
        except Exception as ex:
            print(ex)

        try:
            df_epochs = bad_channel_median(bad_channels_stephanie[sub_], df_epochs)
        except Exception as ex:
            print(ex)

        epoched_dataframes.append([df_epochs, rejects])

    clfSVC = SVC(C=1.0, kernel='poly', gamma='auto_deprecated', coef0=1.0, shrinking=True, probability=False,
                 tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=42)

    for pair in epoched_dataframes:
        X_train, y_train = pair[0], pair[1]
        clfSVC.fit(X_train, y_train)

    test_subjects = dict()
    for sub_test in test_sub_folders:
        sub_id, path_ = sub_test[0], sub_test[1]
        files = os.listdir(Path(path_))
        sub_files = dict()
        for file in files:
            full_path = os.path.join(Path(path_), file)
            if 'epoch' in file:
                sub_files['epoch'] = full_path
            if 'reject' in file:
                sub_files['reject'] = full_path
            elif 'stages' in file:
                sub_files['stage'] = full_path
        test_subjects[sub_id] = sub_files

    epoched_dataframes_ = list()
    for sub_test_ in test_subjects.keys():
        file_path = test_subjects[sub_test_]['epoch']
        mat_reject = test_subjects[sub_test_]['reject']
        mat_stage = test_subjects[sub_test_]['stage']

        files = load_subject_dir(file_path, mat_reject, mat_stage)
        epochs = files['epochs']
        rejects = files['rejects']

        index, scaling_time, scalings = ['epoch', 'time'], 1e3, dict(grad=1e13)
        df = epochs.to_data_frame(
            picks=None, scalings=scalings, scaling_time=scaling_time, index=index)
        df_epochs = df.groupby('epoch').mean()

        try:
            stages = files['stages']
            df_epochs['stage'] = stages
        except Exception as ex:
            print(ex)

        try:
            df_epochs = bad_channel_median(bad_channels_stephanie[file_path], df_epochs)
        except Exception as ex:
            print(ex)

        epoched_dataframes_.append([df_epochs, rejects])

    clf_f1_score, clf_precision_score, clf_recall_score = list(), list(), list()
    for pair_test in epoched_dataframes_:
        X_test, y_test = pair_test[0], pair_test[1]
        y_pred = clfSVC.predict(X_test)
        clf_precision_score += [precision_score(y_test, y_pred)]
        clf_recall_score += [recall_score(y_test, y_pred)]

    print('Recall Scores:')
    for scr in clf_recall_score:
        print(scr)

    print('\nPrecision Scores:')
    for scr in clf_precision_score:
        print(scr)


if __name__ == "__main__":
    main()
