#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='tbear',
      version='0.1.0',
      packages=['tbear'],
      entry_points={
          'console_scripts': [
              'tbear = tbear.__main__:main']
      },
)
