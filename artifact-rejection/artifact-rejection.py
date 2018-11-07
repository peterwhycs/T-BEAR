#!/usr/bin/env python
# coding: utf-8

from utils import *

subjects = list()
stephanie_folder = Path("C:\\Users\\peter\\git\\EEG-artifact-rejection\\artifact-rejection\\eeg-data\\Stephanie")
sub_folders = [os.path.join(stephanie_folder, file) for file in os.listdir(stephanie_folder)]

clfSVC = SVC(C=1.0, gamma='auto', random_state=42)

for sub in sub_folders:
    files = os.listdir(Path(sub))
    temp_sub_files = dict()
    for file in files:
        file_path = os.path.join(Path(sub), file)
        if 'epoch' in file:
            temp_sub_files['epoch'] = file_path
        if 'reject' in file:
             temp_sub_files['reject'] = file_path
        elif 'stages' in file:
            temp_sub_files['stage'] = file_path
    subjects.append(temp_sub_files)

x_train = subjects[:21]
x_test = subjects[22:]

for sub_ in x_train:
    file_path = sub_['epoch']
    mat_reject = sub_['reject']
    mat_stage = sub_['stage']

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

    try:
        stages = files['stages']

    except:
        pass

    df_epochs['stage'] = stages

    df_epochs = df.groupby('epoch').mean()
    X, y = df_epochs.values, rejects
    X, y_true = X, y
    clfSVC.fit(X, y_true)
for sub__ in x_test:
    file_path = sub__['epoch']
    mat_reject = sub__['reject']
    mat_stage = sub__['stage']

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

    try:
        stages = files['stages']
    except:
        pass

    df_epochs['stage'] = stages
    df_epochs = df.groupby('epoch').mean()
    X, y = df_epochs.values, rejects
    X, y_true = X, y
    y_pred = clfSVC.predict(X)

    print("\tRecall: %1.3f" % recall_score(y_true, y_pred))
    print("\tF1: %1.3f\n" % f1_score(y_true, y_pred))
