@echo off

REM Deactivate the previous environment if it existeted.
CALL deactivate 2>nul

REM activate python environment
CALL venv\Scripts\activate

REM Start Jupyter.
jupyter lab

pause
