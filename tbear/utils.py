#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
"""Manageable utility functions that can be independently used by user.
"""
# =============================================================================

from typing import Tuple

import mne
import scipy.io as sio
from helpers import *
from mne import read_epochs_eeglab
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

RANDOM_STATE: int = 42
set_config('MNE_LOGGING_LEVEL', 'ERROR')
set_log_level('ERROR')


def load_epoch_reject(epoch_path: str, reject_path: str) -> Tuple[mne.Epochs, np.ndarray]:
    epoch_file = read_epochs_eeglab(epoch_path)
    reject_file = sio.loadmat(reject_path)['reject'].flatten()
    return epoch_file, reject_file


def reshape_data_2d(dataset: np.ndarray) -> np.ndarray:
    num_epochs, num_features, samples_per_epoch = dataset.shape
    dataset = dataset.reshape((num_epochs, num_features * samples_per_epoch))
    return dataset


def min_max_scale(dataset: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler(feature_range=[0, 1])
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data


def standard_scale(dataset: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)
    return scaled_data


def find_pca_component(scaled_data: np.ndarray, visual: bool = False, random_state: int = RANDOM_STATE) -> int:
    pca = PCA(random_state=random_state).fit(scaled_data)
    explained_var_ratio = np.cumsum(pca.explained_variance_ratio_)
    if visual:
        visualize_pca_component(explained_var_ratio)
    n_components = np.argwhere(explained_var_ratio > 0.90)
    return n_components[0]


def pca_transform(dataset: np.ndarray, n_components: int = 500, random_state: int = RANDOM_STATE) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=random_state)
    transformed_data = pca.fit_transform(dataset)
    return transformed_data


def train_model_ml(clf: Any, subject_arr: np.ndarray, reject_arr: np.ndarray, kfold: bool = False,
                   random_state: int = RANDOM_STATE) -> Any:
    """Train scikit-learn machine learning classifier model. Don't use with neural networks.

    Args:
        clf: A classifier model to be trained.
        subject_arr: The EEG array file of the subject.
        reject_arr: The (epoch) reject array file for the subject array.
        kfold: A toggle to train with kfold. Defaults to False.
        random_state: The random state seed for this iteration. Defaults to 42.

    Returns:
        Trained model.

    """
    if not kfold:
        return train_model_split(clf, subject_arr, reject_arr, random_state=random_state)
    else:
        return train_model_kfold(clf, subject_arr, reject_arr, random_state=random_state)


def predict_model_ml(clf: Any, subject_arr: np.ndarray) -> np.ndarray:
    """Predict which epochs are artifacts using a sciki-learn classifier model.

    Args:
        clf: The trained classifier model.
        subject_arr: The EEG array file of the subject.

    Returns:
        np.ndarray: An array with 1s, epoch is an artifact, and/or 0s.

    """
    return clf.predict(subject_arr)
