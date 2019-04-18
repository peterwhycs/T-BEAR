#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
"""High-level functions to build functions/methods for Class Subject in core.py.
"""
# =============================================================================

from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from helpers import *

RANDOM_STATE: int = 42
set_config('MNE_LOGGING_LEVEL', 'ERROR')
set_log_level('ERROR')


def get_file_name(file_path: str) -> str:
    return Path(file_path).resolve().stem


def load_epoch_reject_files(abs_file_path: str,
                            abs_reject_path: str,
                            resample) -> Tuple[np.ndarray, np.ndarray]:
    epoch_file = load_epoch_set(abs_file_path).resample(resample, npad='auto').get_data()
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

    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f"F1 Score: {f1}")
    print(f"Precision Score: {precision}")
    print(f"Recall Score: {recall}")
    print(f"ROC-AOC Score: {roc_auc}")

    return model


def train_model_kfold(model: Any, X: np.ndarray, y: np.ndarray, n_splits: int = 3,
                      random_state: int = RANDOM_STATE) -> Any:
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=False)
    f1_kfold_scores = []
    pre_kfold_scores = []
    rec_kfold_scores = []
    roc_auc_kfold_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = tf.keras.utils.normalize(X_train, axis=1)
        X_test = tf.keras.utils.normalize(X_test, axis=1)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        f1_kfold_scores.append(f1)

        precision = precision_score(y_test, y_pred)
        pre_kfold_scores.append(precision)

        recall = recall_score(y_test, y_pred)
        rec_kfold_scores.append(recall)

        roc_auc = roc_auc_score(y_test, y_pred)
        roc_auc_kfold_scores.append(roc_auc)

    print(f"F1 Scores: {f1_kfold_scores}")
    print(f"Precision Scores: {pre_kfold_scores}")
    print(f"Recall Scores: {rec_kfold_scores}")
    print(f"ROC-AOC Scores: {roc_auc_kfold_scores}")

    return model
