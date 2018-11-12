#!/usr/bin/env python
# coding: utf-8

import os
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tpot import TPOTClassifier


def resize_reject(reject_array, r=2000):
    """Resizes reject file array to match the number of epochs.
    Arguments:
        reject_array (numpy.ndarray): The freshly converted dataframe.
    Returns:
        numpy.ndarray: Returns a resized list with each element repeated r times respectively.
    """
    repeated_reject_array = np.repeat(reject_array, r)
    return repeated_reject_array


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
    """Cleans dataframe by resetting index, deleting non-essential features, etc.

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


def bad_channel_median(bad_chan, clean_df):
    for channel in bad_chan:
        if channel in clean_df.columns:
            clean_df[channel] = np.median(clean_df[channel])
    return clean_df


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
