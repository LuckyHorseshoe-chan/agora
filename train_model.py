# coding=utf-8
import json
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import re
from copy import deepcopy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle
import nltk
nltk.download('punkt')

def clean_text(text):
    cleaned_text = re.sub('[^А-Яа-яA-Za-z0-9]+', ' ', text)
    cleaned_text = cleaned_text.lower()
    tokens = word_tokenize(cleaned_text)
    text = ' '.join(tokens)
    return text

f = open('agora_hack_products.json')
data = json.load(f)
f.close()

cleaned_data = deepcopy(data)
for i, elem in enumerate(cleaned_data):
  prop_str = ' '.join(elem['props'])
  prop_str = elem['name'] + ' ' + prop_str
  elem['props'] = clean_text(prop_str)
  if elem['is_reference']:
    elem['reference_id'] = elem['product_id']
cleaned_data = pd.DataFrame(cleaned_data)

X = cleaned_data['props']
y = cleaned_data['reference_id'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

vec = CountVectorizer()

X_train_bow = vec.fit_transform(X_train)
X_test_bow = vec.transform(X_test)

f = open("test.txt", "w")
f.write(X_test.to_string())
f.close()


clf = MLPClassifier()
clf.fit(X_train_bow, y_train)

y_pred = clf.predict(X_test_bow)

pickle.dump(clf, open('model.sav', 'wb'))
