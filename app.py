#!/usr/bin/env python
import os
import re
import json
import requests
import pandas as pd
from copy import deepcopy
from train_model import vec
from funcs import modify_props

import pickle

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

from fastapi import FastAPI

from datetime import datetime
import time

app = FastAPI()

@app.post("/id")
def add_product(data: list):
  ts1 = datetime.timestamp(datetime.now())
  cleaned_data = deepcopy(data)
  cleaned_data = modify_props(cleaned_data)
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
      return {"status":"Bad request"}, 400
  ts2 = datetime.timestamp(datetime.now())
  print(ts2-ts1)

  return result
