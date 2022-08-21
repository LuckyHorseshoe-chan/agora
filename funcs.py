from nltk.tokenize import word_tokenize
import re

def clean_text(text):
    cleaned_text = re.sub('[^А-Яа-яA-Za-z0-9]+', ' ', text)
    cleaned_text = cleaned_text.lower()
    tokens = word_tokenize(cleaned_text)
    text = ' '.join(tokens)
    return text
def modify_props(data, ref='n'):
    for i, elem in enumerate(data):
        prop_str = ' '.join(elem['props'])
        prop_str = elem['name'] + ' ' + prop_str
        elem['props'] = clean_text(prop_str)
        if ref == 'y' and elem['is_reference']:
          elem['reference_id'] = elem['product_id']
    return data
