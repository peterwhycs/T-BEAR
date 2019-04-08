#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
"""Helper functions used mostly to build high-level functions in common.py.
"""
# =============================================================================

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io as sio
from mne import read_epochs_eeglab, set_config, set_log_level
from sklearn.model_selection import KFold, train_test_split
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


def reshape_data_2d(dataset: np.ndarray) -> np.ndarray:
    num_epochs, num_features, samples_per_epoch = dataset.shape
    dataset = dataset.reshape((num_epochs, num_features * samples_per_epoch))
    return dataset


def standard_scale(dataset: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data


def min_max_scale(dataset: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler(feature_range=[0, 1])
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data


def visualize_pca_component(expl_var_ratio: np.ndarray) -> None:
    # Plot the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(expl_var_ratio)
    plt.xlabel("Number of Components")
    plt.ylabel("Variance (%)")
    plt.title("Explained Variance")
    plt.show()


def train_model_split(model: Any, X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE) -> Any:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_train)
    print(f"Test Score: {test_score}")
    return model


def train_model_kfold(model: Any, X: np.ndarray, y: np.ndarray, n_splits: int = 3,
                      random_state: int = RANDOM_STATE) -> Any:
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=False)
    kfold_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        kfold_scores.append(model.score(X_test, y_test))
    print(f"Test Scores: {kfold_scores}")
    return model
