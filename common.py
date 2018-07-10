import datetime
import itertools
import os
import pickle
import subprocess
import sys
import time
from datetime import date, timedelta
from itertools import chain, product
from pathlib import Path, PureWindowsPath

import ipywidgets as widgets
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy.io as sio
import tables
from IPython.display import HTML, Javascript, display, display_pretty
from ipykernel.eventloops import register_integration
from ipywidgets import (HTML, Box, Button, Dropdown, FloatText, IntSlider,
						Label, Layout, Textarea, VBox, fixed, interact,
						interact_manual, interactive)
from mne import io
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
