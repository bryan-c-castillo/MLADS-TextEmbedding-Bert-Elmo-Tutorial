#!/bin/sh

# try to deactivate the current environment (if there is one).
deactivate 2>/dev/null

# Create a new python virtual environment.
python3 -m venv venv || exit 1

# Activate the python virtual environment.
. ./venv/bin/activate

# Install deps
python3 -m pip install jupyter jupyterlab pandas 
