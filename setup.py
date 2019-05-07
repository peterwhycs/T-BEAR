#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='T-BEAR',
      version='0.1.0',
      packages=['tbear',],
      description=''
      entry_points={
          'console_scripts': [
              'tbear = tbear.__main__:main']
      },
)
