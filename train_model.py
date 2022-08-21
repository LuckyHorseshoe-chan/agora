# coding=utf-8
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
from funcs import clean_text, modify_props
import nltk
nltk.download('punkt')

f = open('agora_hack_products.json')
data = json.load(f)
f.close()

cleaned_data = deepcopy(data)
cleaned_data = modify_props(cleaned_data, 'y')
cleaned_data = pd.DataFrame(cleaned_data)

X = cleaned_data['props']
y = cleaned_data['reference_id'].values

vec = CountVectorizer()

X_bow = vec.fit_transform(X)

clf = MLPClassifier()
clf.fit(X_bow, y)

pickle.dump(clf, open('model.sav', 'wb'))
