#!/usr/bin/env python
# -*- coding: utf-8 -*-

from importlib import import_module


# Contains information whether a specific package is available
have = {d: False for d in ["numpy", "scipy", "mne", "matplotlib",
                           "pandas", "sklearn", "PyQt5"]}

for key, value in have.items():
    try:
        import_module(key)
    except ModuleNotFoundError:
        pass
    else:
        have[key] = True
