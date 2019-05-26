#!/bin/sh

# Deactivate the previous environment if it existeted.
deactivate 2>/dev/null

# activate python environment
. venv/bin/activate

# Start Jupyter.
jupyter lab
