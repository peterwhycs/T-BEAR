#!/usr/bin/env python
# coding: utf-8

import utilities as utils


def main():
    # Set file paths
    file_path = Path('/home/walker/peterwhy/git/EEG-artifact-rejection/artifact-rejection/eeg-data/Stephanie/Rew_601_rest/Rew_601_rest_bb_epoch.set')
    mat_reject = Path('/home/walker/peterwhy/git/EEG-artifact-rejection/artifact-rejection/eeg-data/Stephanie/Rew_601_rest/Rew_601_rest_reject_rmm.mat')
    mat_stage = Path('/home/walker/peterwhy/git/EEG-artifact-rejection/artifact-rejection/eeg-data/Stephanie/Rew_601_rest/Rew_601_rest_stages.mat')
    files = utils.load_subject_dir(file_path, mat_reject, mat_stage)
    epochs = files['epochs']

    try:
        reject = list(files['reject'])
    except:
        pass

    try:
        stages = list(files['stages'])
    except:
        pass

    print("Cleaning data...")
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

    try: 
        df[['time', 'epoch']] = df[['time', 'epoch']].astype(int)
    except:
        pass

    df_ = df.copy()
    print("Cleaned data successfully!\n")

    # Select values from columns for IForest:
    print("Preparing data for classification...")
    value_columns = list(df.columns)

    try:
        if 'time' in value_columns:
            value_columns.remove('time')
        if 'epoch' in value_columns:
            value_columns.remove('epoch')
    except:
        pass

    df_values = df_[value_columns]
    print("Data prepared successfully!\n")

    # # SVM Classifier:
    # print("Running SVM Classifier..")
    # X_train, y_train = df_values, reject
    # clfSVC = svm.SVC(kernel='linear', C = 1.0)
    # clfSVC.fit(X_train, y_train)

    # IForest:
    print("Running IForest algorithm...")
    X = df_values
    clfIF = IsolationForest(n_estimators=80, max_samples='auto', contamination=0.00001, max_features=1, bootstrap=False, n_jobs=3, random_state=42, verbose=1)
    clfIF.fit(X)

    pred_artifacts = clfIF.predict(X)
    count_artifacts = np.unique(ar=pred_artifacts, return_counts=True)
    index_artifacts = [i for i, x in enumerate(pred_artifacts) if x == -1]

    df_IF = df_.loc[index_artifacts]
    df_IF_epochs = set(df_IF['epoch'])
    print(df_IF_epochs)

    num_artifacts_pair = count_artifacts[1][0]
    num_artifacts = num_artifacts_pair[1][1]

    total_pts = count_artifacts[1][1]
    total_artifacts = np.count_nonzero(reject)
    print("IForest algorithm ran successfully!\n")
    print(f"{num_artifacts} artifacts detected out of {total_artifacts} artifacts total.")


if __name__ == "__main__":
    main()
