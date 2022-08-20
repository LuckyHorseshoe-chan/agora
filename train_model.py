# coding=utf-8
import json
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
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

cleaned_data = data.copy()
samples = []
products = []
for i in range(len(cleaned_data)):
  prop_str = ' '.join(cleaned_data[i]['props'])
  prop_str = cleaned_data[i]['name'] + ' ' + prop_str
  cleaned_data[i]['props'] = clean_text(prop_str)
  if cleaned_data[i]['is_reference']:
    samples.append(cleaned_data[i])
    samples[-1]['reference_id'] = samples[-1]['product_id']
  else:
    products.append(cleaned_data[i])

products_data = pd.DataFrame(products)
samples_data = pd.DataFrame(samples)

X = products_data['props']
y = products_data['reference_id'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train, y_train = X_train.append(samples_data['props']), np.concatenate((y_train, samples_data['reference_id'].values))

vec = CountVectorizer()

X_train_bow = vec.fit_transform(X_train)
X_test_bow = vec.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_bow, y_train)

y_pred = clf.predict(X_test_bow)

pickle.dump(clf, open('model.sav', 'wb'))
