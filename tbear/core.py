#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
"""Class SetFile methods fort T-BEARS tha could be used by public user.
"""
# =============================================================================

from common import *


class SetFile:
    def __init__(self,
                 epoch_file_path: str,
                 reject_file_path: str,
                 name: str = None,
                 scaled: bool = True,
                 pca_transform: bool = False) -> None:
        if not name:
            self.name = ""
        else:
            self.name = name
        epoch_array, reject_array = load_epoch_reject_files(epoch_file_path, reject_file_path)
        self.epoch = epoch_array
        self.reject = reject_array
        self.scaled = scaled
        self.pca_transform = pca_transform

    def pca_transform(self, standard_scaler: bool = True) -> np.ndarray:
        if not self.scaled:
            self.epoch = scale_data(self.epoch, standard=standard_scaler)
        self.epoch = pca_transform(self.epoch)
        self.scaled = self.pca_transform = True
        return self.epoch

    @staticmethod
    def train_model(clf: Any, set_file_array: np.ndarray, reject_file: np.ndarray, kfold_mode: bool = False,
                    random_state: int = RANDOM_STATE) -> Any:
        pass

    @staticmethod
    def predict_reject(clf: Any, set_file_array: np.ndarray) -> np.ndarray:
        return clf.predict(set_file_array)
