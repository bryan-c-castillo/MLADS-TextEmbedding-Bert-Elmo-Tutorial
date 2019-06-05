@echo off

REM Deactivate a possible existing python virtual environment.
CALL deactivate 2>nul

REM Create new python virtual environment under venv
python -m venv venv

CALL venv\Scripts\activate

REM Install packages for jupyter
python -m pip install jupyter jupyterlab pandas 

REM Install packages for ELMo tutorial
python -m pip install update spacy tensorflow tensorflow_hub sklearn xgboost spacy nltk

REM download the spacy corpus for en
python -m spacy download en

REM download wordnet corpus
python -c "import nltk; nltk.download('wordnet')"

REM install dependencies for bert
python pip install bert-tensorflow sklearn tensorflow_hub