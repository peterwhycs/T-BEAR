#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def load_subject_dir(file_path, mat_reject, mat_stage, reject_scaling=False):
    """Loads file paths for EEG data and MATLAB auxiliaries and returns those files.

    Arguments:
        file_path (str): The file path to the .set file.
        mat_stage (str): The file path to the MATLAB file with sleep stages.
        mat_reject (str): The file path to the MATLAB file/array with labels for epoch rejects.

    Returns:
        dict: Returns a dictionary containing all files that did not error.

    Example:
        >>> files = load_subject_dir(file_path, mat_stage, mat_reject)
        >>> files.keys()
        dict_keys(['epochs', 'stages', 'rejects'])
    """
    files = dict()
    found_set, found_sleep, found_reject = True, True, True
    try:
        set_file = mne.io.read_epochs_eeglab(file_path)
        files['epochs'] = set_file
    except:
        set_file = mne.io.read_raw_eeglab(file_path)
        files['epochs'] = set_file
    else:
        pass

    try:
        sleep_file = loadmat(mat_stage)
        sleep = sleep_file['stages'].flatten()
        sleep_ = np.repeat(sleep, 4)
        files['stages'] = sleep_
    except FileNotFoundError:
        found_sleep = False
        pass

    try:
        reject_file = loadmat(mat_reject)
        rejects = reject_file['reject'].flatten()
        if reject_scaling:
            rejects_ = resize_reject(rejects, r=2000)
            files['rejects'] = rejects_
        else:
            files['rejects'] = rejects
    except FileNotFoundError:
        found_reject = False
        pass

    if not found_set:
        print("ERROR: .set file was not found.")
    if not found_sleep:
        print("WARNING: Sleep stages file was not found.")
    if not found_reject:
        print("NOTE: Reject file was not found.")

    return files


def clean_df(df):
    """Cleans dataframe by reseting index, deleting non-essential features, etc.

    Arguments:
        df (pandas.DataFrame): The epochs file converted to a dataframe.

    Returns:
        pandas.DataFrame: Returns a dataframe containing all files that did not error.
    """
    print("Cleaning data...")

    try:
        df = df.drop(['condition'], axis=1)
    except:
        pass

    columns, df = sorted(list(df.columns)), df.reset_index()
    cleaned_columns = ['time']
    if 'epoch' in list(df.columns):
        cleaned_columns += ['epoch']

    cleaned_columns += columns
    df = df[cleaned_columns]

    try:
        df[['time', 'epoch']] = df[['time', 'epoch']].astype(int)
    except:
        pass

    print("Cleaned data successfully!\n")
    return df


def extract_df_values(df):
    df_ = df.copy()
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
    return df_values


def resize_reject(reject_array, r=2000):
    """Resizes reject file array to match the number of epochs.
    Arguments:
        reject_array (numpy.ndarray): The freshly converted dataframe.

    Returns:
        numpy.ndarray: Returns a resized list with each element repeated r times respectively.
    """
    repeated_reject_array = np.repeat(reject_array, r)
    return repeated_reject_array


def reject_epochs(reject_index, epochs, df):
    """Converts sampled rejection array into epoched rejection array.

        Arguments:
            reject_index (numpy.ndarray): An array of the indices of sample points that were flagged as artifacts.
            epochs (int): The total number of epochs.
            df (pandas.DataFrame): The dataframe prior to machine learning algorithm.

        Returns:
            numpy.ndarray: Returns an array with y_pred, rejected sample points, mapped to corresponding epochs.

        Examples:
            >>> import numpy as np
            >>> y_pred = np.array([1, 1, 1, 0, ... , 0])  # Sample y_pred where 1s represent rejections
            >>> y_pred_epochs = reject_epochs(y_pred, reject_index, num_epochs, df)
            >>> y_pred_epochs  # All sample points belong to the first epoch, or epoch 0
            np.array([1, 0, ... , 0])
    """
    epoch_array, epoch_index = np.zeros(epochs), np.asarray(sorted(set(df.loc[reject_index, 'epoch'].values)))
    for index in epoch_index:
        epoch_array[index] = 1
    return epoch_array


def run_IForest(X, y, df):
    print("Running IForest algorithm...")
    clfIF = IsolationForest(n_estimators=100, max_samples=1, contamination=0.003,
                            max_features=1.0, bootstrap=False, n_jobs=2, random_state=42, verbose=0)
    clfIF.fit(X)
    pred_artifacts = clfIF.predict(X)
    index_artifacts = [i for i, x in enumerate(pred_artifacts) if x == -1]
    df_IF = df.loc[index_artifacts]
    print("IForest algorithm ran successfully!\n")
    return df_IF


# def reject_arr(df, df_IF):
#     df['artifact'] = 0
#     reject_index = sorted(list(df_IF.index))
#     df_ = df.loc[reject_index,'artifact'] = 1
#     return list(df_['artifact'])


def run_SVM(epoch_3d, rejects):
    print("Running SVM Classifier..")
    X, y = epoch_3d, rejects
    clfSVC = SVC(C=1.0, andom_state=42)
    clfSVC.fit(X, y)
    y_pred = clfSVC.predict(X)
    acc_score = accuracy_score(y, y_pred, normalize=True, sample_weight=None)
    print('Accuracy Score (Normalized):', acc_score)
    return y_pred
