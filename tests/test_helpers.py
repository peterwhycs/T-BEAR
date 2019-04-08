#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from context import *
from helpers import *


class TestHelpers(unittest.TestCase):
    """Basic low-level function test cases."""

    def setUp(self) -> None:
        self.epoch_path = os.path.realpath(os.path.join("data", "test_file_epoch.set"))
        self.file = read_epochs_eeglab(self.epoch_path).get_data()[:2324]
        self.reject_path = os.path.realpath(os.path.join("data", "test_reject.mat"))

    def test_load_epoch(self):
        file_ = load_epoch_set(self.epoch_path)[:2324]
        self.assertSequenceEqual(file_.tolist(), self.file.tolist())

    def test_load_reject(self):
        reject_file = load_reject_mat(self.reject_path)
        self.assertTrue(type(reject_file), np.ndarray)

    def test_reshape_2d(self):
        reshaped_arr = reshape_data_2d(self.file)
        self.assertEqual(len(reshaped_arr.shape), 2)

    def tearDown(self):
        self.epoch_path = None
        self.file = None
        self.reject_path = None


if __name__ == "__main__":
    unittest.main()
