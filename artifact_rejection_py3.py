from pathlib import Path, PureWindowsPath

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne import io
from sklearn.ensemble import IsolationForest

# Set 'file_path':
file_path = str(Path(r'ANTT_104/ANTT_104_SR_bb_epoch.set'))

# Load 'file_path':
try:
    epochs = mne.io.read_epochs_eeglab(file_path)
except:
    epochs = mne.io.read_raw_eeglab(file_path)

# Convert to and clean DataFrame:
df = epochs.to_data_frame()
columns, df = sorted(list(df.columns)), df.reset_index()

try: 
    df = df.drop(['condition'], axis=1)
except:
    pass

cleaned_columns = ['time']
if 'epoch' in list(df.columns):
    cleaned_columns += ['epoch']
cleaned_columns += columns

df = df[cleaned_columns]
df_ = df.copy()

# Select values from columns for IForest:
value_columns = list(df.columns)

try:
    if 'time' in value_columns:
        value_columns.remove('time')
    if 'epoch' in value_columns:
        value_columns.remove('epoch')
except:
    pass

df_values = df_[value_columns]

# Run IForest:
X = df_values
clfIF = IsolationForest(random_state=42, contamination=0.00001, n_jobs=2)
clfIF.fit(X)
pred_train, pred_test = clfIF.predict(X), clfIF.predict(X)
count_train, count_test = np.unique(ar=pred_train, return_counts=True), np.unique(ar=pred_test, return_counts=True)
index_train, index_test = [i for i,x in enumerate(pred_train) if x == -1] , [i for i,x in enumerate(pred_test) if x == -1]
df_IF = df_.loc[index_test]
num_anomalies = count_train[1][0], count_test[1][0]; total_pts = count_train[1][1], count_test[1][1]

print("Number of anomalies: ", num_anomalies)
print("Total points:", total_pts)

