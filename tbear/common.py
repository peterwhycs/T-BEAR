#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
"""High-level functions to build functions/methods for Class Subject in core.py.
"""
# =============================================================================

from pathlib import Path
from typing import Tuple

from sklearn.decomposition import PCA

from helpers import *

RANDOM_STATE: int = 42
set_config('MNE_LOGGING_LEVEL', 'ERROR')
set_log_level('ERROR')


def get_file_name(file_path: str) -> str:
    return Path(file_path).resolve().stem


def load_epoch_reject_files(abs_file_path: str,
                            abs_reject_path: str) -> Tuple[np.ndarray, np.ndarray]:
    epoch_file = reshape_data_2d(load_epoch_set(abs_file_path))
    reject_file = load_reject_mat(abs_reject_path)
    return epoch_file, reject_file


def scale_data(dataset: np.ndarray, standard: bool = True) -> np.ndarray:
    if standard:
        scaled_data = standard_scale(dataset)
    else:
        scaled_data = min_max_scale(dataset)
    return scaled_data


def find_pca_component(scaled_data: np.ndarray,
                       visual: bool = False,
                       random_state: int = RANDOM_STATE) -> int:
    pca = PCA(random_state=random_state).fit(scaled_data)
    explained_var_ratio = np.cumsum(pca.explained_variance_ratio_)
    if visual:
        visualize_pca_component(explained_var_ratio)
    n_components = np.argwhere(explained_var_ratio > 0.90)
    return n_components[0]


def pca_transform(dataset: np.ndarray,
                  n_components: int = 500,
                  random_state: int = RANDOM_STATE) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=random_state)
    transformed_data = pca.fit_transform(dataset)
    return transformed_data
