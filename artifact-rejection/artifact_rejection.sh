#!/bin/bash
DATADIR=”$HOME/DATA”
DATASET=”SUBJECT_101”
echo “my data is here: $DATADIR/$DATASET”
python3 artifact-rejection.py
