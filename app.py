#!/usr/bin/env python
import os
import re
import json
import requests
import pandas as pd

import joblib
import gzip

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

from flask import Flask, request, jsonify
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

app = Flask(__name__)

# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/", methods = ["POST"])
def add_product():
  with app.app_context():
    if not request.get_json() or not "is_reference" in request.json:
      return jsonify({"status":"Bad request"}), 400
    is_reference = request.get_json()["is_reference"]
    if is_reference:
        return jsonify(template='None')

    if not "name" in request.json:
      return jsonify({"status":"Bad request"}), 400
    name = request.get_json()["name"]
    new_name = clean_text(name)

    if not "props" in request.json:
      return jsonify({"status":"Bad request"}), 400
    props = request.get_json()["props"]

#    new_props = ' '.join(props)
    props = new_name + ' ' + clean_text(props)

    try:
      vec = CountVectorizer()
      text = vec.fit_transform([props])

      model = joblib.load(open('agora.dat', 'rb'))
      result = compare(model, text)
    except ValueError:
      return jsonify({"status":"Bad request"}), 400

    message = {'template': result}

    return jsonify(message)
