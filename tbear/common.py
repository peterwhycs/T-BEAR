#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
"""High-level functions to build functions/methods for Class Subject in core.py.
"""
# =============================================================================

from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from helpers import *

RANDOM_STATE: int = 42
set_config('MNE_LOGGING_LEVEL', 'ERROR')
set_log_level('ERROR')


def get_file_name(file_path: str) -> str:
    return Path(file_path).resolve().stem


def load_epoch_reject_files(abs_file_path: str,
                            abs_reject_path: str) -> Tuple[np.ndarray, np.ndarray]:
    epoch_file = load_epoch_set(abs_file_path)
    reject_file = load_reject_mat(abs_reject_path)
    return epoch_file, reject_file


def scale_data(dataset: np.ndarray, standard: bool = True) -> np.ndarray:
    if standard:
        scaled_data = standard_scale(dataset)
    else:
        scaled_data = min_max_scale(dataset)
    return scaled_data


def visualize_pca_component(expl_var_ratio: np.ndarray) -> None:
    # Plot the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(expl_var_ratio)
    plt.xlabel("Number of Components")
    plt.ylabel("Variance (%)")
    plt.title("Explained Variance")
    plt.show()


def find_pca_component(scaled_data: np.ndarray,
                       visual: bool = False,
                       random_state: int = RANDOM_STATE) -> int:
    pca = PCA(random_state=random_state).fit(scaled_data)
    explained_var_ratio = np.cumsum(pca.explained_variance_ratio_)
    if visual:
        visualize_pca_component(explained_var_ratio)
    n_components = np.argwhere(explained_var_ratio > 0.90)
    return n_components[0]


def pca_transform_default(dataset: np.ndarray,
                          n_components: int = 500,
                          random_state: int = RANDOM_STATE) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=random_state)
    transformed_data = pca.fit_transform(dataset)
    return transformed_data


def train_model_split(model: Any, X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_STATE) -> Any:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"Test Score: {f1}")
    return model


def train_model_kfold(model: Any, X: np.ndarray, y: np.ndarray, n_splits: int = 3,
                      random_state: int = RANDOM_STATE) -> Any:
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=False)
    kfold_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        kfold_scores.append(f1)
    print(f"Test Scores: {f1}")
    return model
