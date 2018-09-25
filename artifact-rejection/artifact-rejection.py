#!/usr/bin/env python
# coding: utf-8

from utils import *

warnings.filterwarnings('ignore')


def main():
    # Set file paths
    file_path = Path('eeg-data/601/Rew_601_rest_bb_epoch.set')
    mat_reject = Path('eeg-data/601/Rew_601_rest_reject_rmm.mat')
    mat_stage = Path('eeg-data/601/Rew_601_rest_stages.mat')

    win_file_path = PureWindowsPath(file_path)
    win_mat_reject = PureWindowsPath(mat_reject)
    win_mat_stage = PureWindowsPath(mat_stage)

    files = load_subject_dir(file_path, mat_reject, mat_stage)
    epochs = files['epochs']

    try:
        reject = files['reject']
    except:
        pass

    try:
        sleep_stages = files['stages']
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
    df_ = df.copy()
    print("Cleaned data successfully!\n")

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
    print("Data prepared successfully!\n")

    # Run IForest:
    print("Running IForest algorithm...")
    X = df_values
    clfIF = IsolationForest(random_state=42, contamination=0.00001, n_jobs=3)
    clfIF.fit(X)
    pred_train, pred_test = clfIF.predict(X), clfIF.predict(X)
    count_train, count_test = np.unique(
        ar=pred_train, return_counts=True), np.unique(ar=pred_test, return_counts=True)
    index_train, index_test = [i for i, x in enumerate(pred_train) if x == -1], [i for i, x in enumerate(pred_test) if
                                                                                 x == -1]
    df_IF = df_.loc[index_test]
    num_artifacts_pair = count_train[1][0], count_test[1][0]
    num_artifacts = num_artifacts_pair[1]
    total_pts = count_train[1][1], count_test[1][1]
    total_artifacts = np.count_nonzero(reject)
    accuracy_percent = num_artifacts / total_artifacts * 100
    print("IForest algorithm ran successfully!\n")

    print(f"Performance: {accuracy_percent}%")
    print(f"{num_artifacts} artifacts detected out of {total_artifacts} artifacts total.")


if __name__ == "__main__":
    main()
