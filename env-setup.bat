

REM Deactivate a possible existing python virtual environment.
CALL deactivate 2>nul

REM Create new python virtual environment under venv
python -m venv venv

CALL venv\Scripts\activate

python -m pip install jupyter jupyterlab pandas 


