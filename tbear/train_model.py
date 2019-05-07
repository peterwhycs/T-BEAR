#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
"""Script to train ML or DL model.
"""
# =============================================================================

import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils import train_model_ml

RANDOM_STATE: int = 42
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR_PATH = os.path.realpath(
    os.path.join(DIR_PATH, '..', 'data', 'epoched-files'))  # The directory with subjects' EEG .set files
REJ_DIR_PATH = os.path.realpath(
    os.path.join(DIR_PATH, '..', 'data', 'reject-files'))  # The directory with subjects' reject files


def train_baseline_model(epochs, reject_arr, kfold: bool = True, n_jobs: int = -1,
                         random_state: int = RANDOM_STATE):
    clf = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=random_state)
    train_model_ml(clf=clf, subject_arr=np.array([]), reject_arr=reject_arr, kfold=kfold, random_state=random_state)


if __name__ == '__main__':
    pass
