# MLADS-TextEmbedding-Bert-Elmo-Tutorial

Examples of using Bert and ELMo.

## Warning

These notebooks use data from IMDB. Some of the notebooks display randomly selected reviews and many of those reviews or movies may be considered offensive.

## Environment and Setup

These notebooks use Python3 and was tested using Python 3.6. The notebooks were tested on Ubuntu and Windows. The notebooks were tested using Azure Machine Learning service workspace Notebook VMs.

Scripts are provided to run the notebooks locally using Python virtual environments as well. See the following scripts to run locally.

* env-setup.sh, env-setup.bat
* start-jupyter.sh, start-jupyter.bat

## Original Examples

The notebooks in this repository referenced the following code:

* https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb

## Running

<b>Prerequisite</b>: You have a running jupyter instances, either locally or using a cloud hosted notebook VM.

### Step 1 - Dependencies

01.install-dependencies

Run the install-dependencies notebook to pull in the required libraries and download text corpora.

Note: You may choose to do this with commands in your terminal as well.

### Step 2 - Acquire Data

02.acquire-data

The acquire-data notebook will download data originally from IMDB and will parse out postivies and negative movie reviews. A pandas data frame will be saved containing the reviews, whether they were positive or negatives, and the IMDB movie id.

### Step 3 - Extract ELMo Embeddings

03.elmo-embedding

This notebook will use a pre-trianed ELMo model to extract the text embeddings for the moview revies. The embeddings will be added to a data frame and saved.

## Step 4 - Classification On ELMo Embeddings

04.elmo-embedding-classification

This notebook uses the embedding extracted in step 3 as a feature in an XGBoost classifier.

## Step 5 - Bert

05.bert

This notebook uses Bert to classify IMDB moview reviews. In the ELMo example, the embeddings were used in a different model. In this notebook a new classification layer is added to the Bert model and trained.

The code for adding a layer to Bert, training, testing, and getting predictions was factored into into a separate python file, bert_classifier.py.

