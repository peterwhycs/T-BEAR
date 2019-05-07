#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
"""Architectural code for bootstrapping components and developer ergonomics.
"""
# =============================================================================

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import normalize
from mne import set_config, set_log_level
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split

RANDOM_STATE: int = 42
set_config('MNE_LOGGING_LEVEL', 'ERROR')
set_log_level('ERROR')


def get_name(file_path: str) -> str:
    path_str = Path(file_path).resolve().stem
    path_str = path_str.replace('_epoch', '')
    path_str = path_str.replace('epoch', '')
    path_str = path_str.replace('_reject', '')
    path_str = path_str.replace('reject', '')
    return path_str


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

    X_train = normalize(X_train, axis=1)
    X_test = normalize(X_test, axis=1)

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

        X_train = normalize(X_train, axis=1)
        X_test = normalize(X_test, axis=1)

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
