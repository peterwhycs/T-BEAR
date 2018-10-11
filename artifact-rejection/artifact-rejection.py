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

    try:
        stages = list(files['stages'])
    except:
        pass

    index, scaling_time, scalings = ['epoch', 'time'], 1e3, dict(grad=1e13)
    df_read = epochs.to_data_frame(
        picks=None, scalings=scalings, scaling_time=scaling_time, index=index)
    df = clean_df(df_read)
    df_ = df.copy()

    # Select values from dataframe
    df_values = extract_df_values(df)

    # SVM Classifier:
    print("Running SVM Classifier..")
    X_train, y_train = df_values, reject
    clfSVC = svm.SVC(kernel='linear', C=1.0)
    clfSVC.fit(X_train, y_train)
    y_pred = clfSVC.predict(X_train)
    acc_score = accuracy_score(y_train, y_pred)
    print('Accuracy Score:', acc_score)

    # # IForest:
    # print("Running IForest algorithm...")
    # X = df_values
    # clfIF = IsolationForest(n_estimators=80, max_samples='auto', contamination=0.001,
    #                         bootstrap=False, n_jobs=3, random_state=42, verbose=1)
    # clfIF.fit(X)

    # pred_artifacts = clfIF.predict(X)
    # count_artifacts = np.unique(ar=pred_artifacts, return_counts=True)
    # index_artifacts = [i for i, x in enumerate(pred_artifacts) if x == -1]

    # df_IF = df_.loc[index_artifacts]
    # df_IF_epochs = set(df_IF['epoch'])
    # print(df_IF_epochs)

    # num_artifacts_pair = count_artifacts[1][0]
    # num_artifacts = num_artifacts_pair[1][1]

    # total_pts = count_artifacts[1][1]
    # total_artifacts = np.count_nonzero(reject)
    # print("IForest algorithm ran successfully!\n")
    # print(f"{num_artifacts} artifacts detected out of {total_artifacts} artifacts total.")


if __name__ == "__main__":
    main()
