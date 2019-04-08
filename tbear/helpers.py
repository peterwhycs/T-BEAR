#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
"""Helper functions used mostly to build high-level functions in common.py.
"""
# =============================================================================

import numpy as np
import scipy
import scipy.io as sio
from mne import read_epochs_eeglab, set_config, set_log_level
from sklearn.preprocessing import MinMaxScaler, StandardScaler

RANDOM_STATE: int = 42
set_config('MNE_LOGGING_LEVEL', 'ERROR')
set_log_level('ERROR')


def load_epoch_set(abs_epoch_path: str) -> np.ndarray:
    epoch_file = read_epochs_eeglab(abs_epoch_path)
    file = epoch_file.get_data()
    return file


def load_reject_mat(abs_reject_path: str) -> np.ndarray:
    reject_file = scipy.io.loadmat(abs_reject_path)["reject"].flatten()
    return reject_file


def min_max_scale(dataset: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler(feature_range=[0, 1])
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data


def reshape_data_2d(dataset: np.ndarray) -> np.ndarray:
    num_epochs, num_features, samples_per_epoch = dataset.shape
    dataset = dataset.reshape((num_epochs, num_features * samples_per_epoch))
    return dataset


def standard_scale(dataset: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data
