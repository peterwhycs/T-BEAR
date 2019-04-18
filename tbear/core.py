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
                 resample: int = 200,
                 scaled: bool = False) -> None:
        if not name:
            self.name = get_file_name(subject_file_path)
        else:
            self.name = name
        subject_arr, reject_arr = load_epoch_reject_files(subject_file_path, reject_file_path, resample=resample)
        self.subject = subject_arr
        self.reject = reject_arr
        self.scaled = scaled

    def __str__(self) -> str:
        return str(self.name)

    def pca_transform(self, standard_scaler: bool = True) -> np.ndarray:
        self.subject = reshape_data_2d(self.subject)
        if not self.scaled:
            self.subject = scale_data(self.subject, standard=standard_scaler)
            self.scaled = True
        self.subject = pca_transform_default(self.subject)

        return self.subject

    @classmethod
    def add_to_group(cls, subject):
        Subject.subject_group.append(subject.subject)
        Subject.reject_files.append(subject.reject)

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
