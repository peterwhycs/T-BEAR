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

        self.epoch_path = subject_file_path
        self.reject_path = reject_file_path

    def __repr__(self) -> str:
        return 'Subject(' + self.name + ')'

    def __str__(self) -> str:
        return str(self.name)

    @staticmethod
    def dir_to_subs(dir_epoch: str, dir_rej: str) -> List['Subject']:
        group = []
        epoch_files = sorted(os.listdir(dir_epoch))
        reject_files = sorted(os.listdir(dir_rej))
        for e, r in zip(epoch_files, reject_files):
            epoch = os.path.abspath(os.path.join(dir_epoch, e))
            reject = os.path.abspath(os.path.join(dir_rej, r))
            group.append(Subject(epoch, reject))
        return group
