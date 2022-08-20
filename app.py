#!/usr/bin/env python
import os
import re
import json
import requests
import pandas as pd
from copy import deepcopy
from train_model import vec

import pickle

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

from flask import Flask, request, jsonify
from fastapi import FastAPI

from datetime import datetime
import time
# , make_response

# from flask_cors import CORS, cross_origin


def compare(model, X):
  prediction = model.predict(X)
  return prediction


def clean_text(text):
  cleaned_text = re.sub('[^А-Яа-яA-Za-z0-9]+', ' ', text)
  cleaned_text = cleaned_text.lower()
  tokens = word_tokenize(cleaned_text)
  text = ' '.join(tokens)
  return text

app = FastAPI()

# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

@app.post("/id")
def add_product(data: list):
  ts1 = datetime.timestamp(datetime.now())
  cleaned_data = deepcopy(data)
  for i, elem in enumerate(cleaned_data):
    prop_str = ' '.join(elem['props'])
    prop_str = elem['name'] + ' ' + prop_str
    elem['props'] = clean_text(prop_str)
  cleaned_data = pd.DataFrame(cleaned_data)
  X = cleaned_data['props']
  try:
      X_bow = vec.transform(X)

      model = pickle.load(open('model.sav', 'rb'))
      y_pred = model.predict(X_bow)
      probas = model.predict_proba(X_bow)
      result = []
      for i, elem in enumerate(probas):
        if max(elem) < 0.5:
          y_pred[i] = None
        result.append({"id": cleaned_data['id'][i], "reference_id": y_pred[i]})
        
  except ValueError:
      return jsonify({"status":"Bad request"}), 400
  ts2 = datetime.timestamp(datetime.now())
  print(ts2-ts1)

  return result
