#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
"""Class Subject methods fort T-BEARS tha could be used by public user.
"""
# =============================================================================

from common import *

RANDOM_STATE: int = 42
set_config('MNE_LOGGING_LEVEL', 'ERROR')
set_log_level('ERROR')


class Subject:
    subject_group = []
    reject_files = []

    def __init__(self,
                 subject_file_path: str,
                 reject_file_path: str,
                 name: str = None,
                 scaled: bool = False) -> None:
        if not name:
            self.name = ""
        else:
            self.name = name
        subject_arr, reject_arr = load_epoch_reject_files(subject_file_path, reject_file_path)
        self.subject = subject_arr
        self.reject = reject_arr
        self.scaled = scaled

    def pca_transform(self, standard_scaler: bool = True) -> np.ndarray:
        if not self.scaled:
            self.subject = scale_data(self.subject, standard=standard_scaler)
        self.subject = pca_transform(self.subject)
        self.scaled = True
        return self.subject

    def add_to_group(self):
        Subject.subject_group.append(self.subject)
        Subject.reject_files.append(self.reject)

    @staticmethod
    def train_model(clf: Any, subject_arr: np.ndarray, reject_arr: np.ndarray,
                    kfold: bool = False, random_state: int = RANDOM_STATE) -> Any:
        if not kfold:
            return train_model_split(clf, subject_arr, reject_arr, random_state=random_state)
        else:
            return train_model_kfold(clf, subject_arr, reject_arr, random_state=random_state)

    @staticmethod
    def predict_model(clf: Any, subject_arr: np.ndarray) -> np.ndarray:
        return clf.predict(subject_arr)
