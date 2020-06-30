import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
import pickle
import re
from janome.tokenizer import Tokenizer
from aozora import Aozora
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, render_template
aozora = Aozora("wagahaiwa_nekodearu.txt")
pndicfname = "./pn_ja.dic"
app = Flask(__name__)

def readpndic(filename):
    with open(filename, "r") as dicfile:
        items = dicfile.read().splitlines()
    return {u.split(':')[0]: float(u.split(':')[3]) for u in items}

def get_string():
    string = '\n'.join(aozora.read())
    string = re.sub('　', '', string)
    string = re.split('。(?!」)|\n', re.sub('　', '', string))
    while '' in string:  string.remove('')
    return string

def get_sentiment(string):
    pndic = readpndic(pndicfname)
    t=Tokenizer()
    sentensewordlist = [ \
        [v.surface for v in t.tokenize(sentense) \
        if (v.part_of_speech.split(',')[0] in ['名詞','形容','動詞','副詞'])]  \
        for sentense in string[:10]]
    sentensewordlist = np.asarray(sentensewordlist)

    sentiment_values = list()
    for sentense in sentensewordlist[[0, 3, 4, 5, 6, 7]]:
        pnlist = [pndic.get(v) for v in sentense if pndic.get(v)!=None]
        sentiment_values.append(np.sum(pnlist))
    return sentiment_values[1:]

def cos_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def get_tfidf(string):
    wakatilist = []
    t = Tokenizer()
    for sentense in string[:10]:
        tokens=[token.surface for token in t.tokenize(sentense)]
        wakatilist.append(" ".join(tokens))

    wakatilist = np.array(wakatilist)
    wakatilist = wakatilist[[0, 3, 4, 5, 6, 7]]
    vectorizer = TfidfVectorizer(use_idf=True, norm=None, token_pattern=u'(?u)\\b\\w+\\b')
    tfidf = vectorizer.fit_transform(wakatilist)
    tfidf_vector = tfidf.toarray()
    cosin_similar = [cos_sim(tfidf_vector[0], tfidf_vector[i]) for i in range(1, len(tfidf_vector))]
    return cosin_similar

string = get_string()
tfidf = get_tfidf(string)
sentiment = get_sentiment(string)

@app.route('/')
def index():
    return render_template('index.html', graph_label=list(range(1, len(tfidf)+1)),
    tfidf_value=tfidf, sentiment_value=sentiment)

if __name__ == '__main__':
    app.debug = True
    app.run()
