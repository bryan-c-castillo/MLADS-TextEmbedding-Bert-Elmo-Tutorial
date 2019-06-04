#!/bin/sh
# A script to convert a python notebook to a script and run it in the background.

jupyter_cmd=/anaconda/envs/py36/bin/jupyter
python_cmd=/anaconda/envs/py36/bin/python
nb_file="$1"
log_dir="logs"

if [[ "$nb_file" == "" ]]; then
    echo "[ERROR] Usage $0 <python_notebook>" >&2
    exit 1
elif [[ ! -f "$nb_file" ]]; then
    echo "[ERROR] $nb_file does not exist." >&2
    exit 1
fi

py_file="$(echo "$nb_file" | sed -e 's/.ipynb/.py/')"
log_file="$(echo "$nb_file" | sed -e 's/.ipynb/.log/')"

"$jupyter_cmd" nbconvert --to script "$nb_file" || exit 1
mkdir -p logs

"$python_cmd" "$py_file" > "$log_dir/$log_file" 2>&1 &
echo "Running $py_file under process $!."
echo "Output in $log_dir/$log_file"