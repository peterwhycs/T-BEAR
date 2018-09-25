import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore')


def load_subject_files(file_path, mat_stage, mat_reject):
    # Load sleep stages & other files:
    files = list()
    found_set, found_sleep, found_reject = True, True, True
    try:
        epochs = mne.io.read_epochs_eeglab(file_path)
    except:
        epochs = mne.io.read_raw_eeglab(file_path)
    else:
        pass

    try:
        sleep_file = loadmat(mat_stage)
        sleep = sleep_file['stages'].flatten()
        files += sleep
    except FileNotFoundError:
        found_sleep = False
        pass

    try:
        reject_file = loadmat(mat_reject)
        reject = reject_file['reject'].flatten()
        files += reject
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
