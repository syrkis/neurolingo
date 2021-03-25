import torch
import json
import nltk
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize
import gensim
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sys import exit


def reader(file):
    data = []
    with open(f'../data/{file}', 'r') as file:
        for line in file.readlines()[:]:
            data.append(json.loads(line))
    return data


dev, test, train = [reader(file) for file in ['music_reviews_dev.json',
                                              'music_reviews_test_masked.json',
                                              'music_reviews_train.json']]
print("DATA LOADED")
def tokenize(file):
        X = []
        y = []
        for review in file:
            if 'reviewText' in review.keys():
                X.append(word_tokenize(review['reviewText']))
                if review['sentiment'] == 'positive':
                    y.append(1)
                elif review['sentiment'] == 'negative':
                    y.append(0)
                else:
                    y.append(-1)
        return X, y

dev_tok, test_tok, train_tok = [tokenize(file) for file in [dev, test, train]]
dev_X, dev_y, test_X, test_y, train_X, train_y = dev_tok[0], dev_tok[1], test_tok[0], test_tok[1], train_tok[0], train_tok[1]

print("LOADING EMBEDDINGS")
model = gensim.models.KeyedVectors.load_word2vec_format('../data/twitter.bin', binary=True)
vocab = set(model.vocab.keys())
print("EMBEDDINGS LOADED")

def embed(data):
    out = []
    for review in tqdm(data):
        tmp = []
        for word in review:
            tmp.append(model[word]) if word in vocab else tmp.append(model['<U>'])
        t = np.mean(np.array(tmp),axis=0)
        out.append(t)
    return out

print("STARTED HERE")

dev_X, test_X, train_X = [embed(data) for data in [dev_X, test_X, train_X]]

print("TRAINING")
clf = make_pipeline(StandardScaler(),SVC(max_iter=100))
clf.fit(train_X, train_y)
p = clf.predict(dev_X)

print(f1_score(dev_y, p))
print(clf.predict(test_X))
pickle.dump(clf,open(f"svm_model","wb"))
