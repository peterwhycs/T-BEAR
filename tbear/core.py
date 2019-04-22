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
    def train_model_ml(clf: Any, subject_arr: np.ndarray, reject_arr: np.ndarray,
                       kfold: bool = False, random_state: int = RANDOM_STATE) -> Any:
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

    @staticmethod
    def predict_model_ml(clf: Any, subject_arr: np.ndarray) -> np.ndarray:
        """Predict which epochs are artifacts using a sciki-learn classifier model.

        Args:
            clf: The trained classifier model.
            subject_arr: The EEG array file of the subject.

        Returns:
            np.ndarray: An array with 1s, epoch is an artifact, and/or 0s.

        """
        return clf.predict(subject_arr)
