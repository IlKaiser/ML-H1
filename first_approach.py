import pandas as pd
from networkx.readwrite import json_graph
import numpy as np

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import HashingVectorizer,CountVectorizer,TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.extmath import density
from sklearn import metrics

import math

## Train and test model with first approach

def train_and_test_model(algorithm=LinearSVC(),verbose=True,vect_type="hash"):

    x_data=pd.read_json('./dataset/noduplicatedataset.json',lines=True)
   
    X_all =  x_data.loc[:,'lista_asm']
    Y_all =  x_data.loc[:,'semantic']

    #Filter General Purpose register name
    registers = [ 'rax','eax','ax',
                   'rbx','ebx','bx',
                   'rcx','ecx','cx',
                   'rdx','edx','dx',
                   'rbp','rbp','bp',
                   'rdi','edi','di',
                   'rsi','esi', 'si']

    if vect_type == "hash":                                  
        vectorizer = HashingVectorizer(stop_words=registers)
        X_all = vectorizer.transform(X_all)
    elif vect_type == "count":
        vectorizer = CountVectorizer(stop_words=registers)
        X_all = vectorizer.fit_transform(X_all)
    elif vect_type == "tfid":
        vectorizer = TfidfVectorizer(stop_words=registers)
        X_all = vectorizer.fit_transform(X_all)
    else:
        raise RuntimeError("Supported Vectorizer Types : hash,count,tfid")

    X_train, X_test, y_train, y_test = train_test_split(X_all, Y_all, test_size=0.3, 
                                                    random_state=42)
    clf     = algorithm
    clf.fit(X_train,y_train)
    pred    = clf.predict(X_test)
    score   = metrics.accuracy_score(y_test, pred)
    if verbose:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,))
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))
    return clf,score,vectorizer


## Evaluate myster set with the trained model

def evaluate_mystery_set(clf,vect=None):
    x_data=pd.read_json('./dataset/blindtest.json',lines=True)
    
    X_test = vect.transform(x_data.loc[:,'lista_asm'])

    y = clf.predict(X_test)
    return y

