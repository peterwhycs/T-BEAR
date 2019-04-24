#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
"""Class Subject methods for T-BEARS that could be used by public user.
"""
# =============================================================================

import os
from typing import List

from utils import *


class Subject:

    def __init__(self, subject_file_path: str, reject_file_path: str, name: str = None) -> None:
        if not name:
            self.name = get_name(subject_file_path)
        else:
            self.name = name
        epoch_file, reject_file = load_epoch_reject(subject_file_path, reject_file_path)
        self.epochs = epoch_file
        self.reject = reject_file

    def __str__(self) -> str:
        return str(self.name)


def dir_to_subjects(dir_epochs: str, dir_rejects) -> List[Subject]:
    dir_lst = []
    epoch_files = sorted(os.listdir(dir_epochs))
    reject_files = sorted(os.listdir(dir_rejects))

    for e, r in zip(epoch_files, reject_files):
        epoch = os.path.abspath(os.path.join(dir_epochs, e))
        reject = os.path.abspath(os.path.join(dir_rejects, r))
        dir_lst.append(Subject(epoch, reject))

    return dir_lst
