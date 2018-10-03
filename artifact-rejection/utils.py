import warnings
from pathlib import Path, PureWindowsPath
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib import style
from scipy.io import loadmat
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
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
        files['reject'] = rejects
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
