import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib import style
from scipy.io import loadmat
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report
style.use("ggplot")


def load_subject_dir(file_path, mat_reject, mat_stage):
    """Loads file paths for EEG data and MATLAB auxiliaries and returns those files.

    Arguments:
        file_path (str): The file path to the .set file.
        mat_stage (str): The file path to the MATLAB file with sleep stages.
        mat_reject (str): The file path to the MATLAB file/array with labels for epoch rejects.

    Returns:
        dict: Returns a dictionary containing all files that did not error.

    Examples:
        >>> files = load_subject_dir(file_path, mat_stage, mat_reject)
        >>> files.keys()
        dict_keys(['epochs', 'stages', 'reject'])
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
        files['stages'] = sleep
    except FileNotFoundError:
        found_sleep = False
        pass

    try:
        reject_file = loadmat(mat_reject)
        rejects = reject_file['reject'].flatten()
        rejects_ = resize_reject(rejects)
        files['reject'] = rejects_
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
        df (DataFrame): The freshly converted dataframe.

    Returns:
        DataFrame: Returns a dataframe containing all files that did not error.
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


def resize_reject(reject_array, r=2000):
    repeated_reject_array = np.repeat(reject_array, r)
    return repeated_reject_array


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
