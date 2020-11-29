import pandas as pd
import networkx as nx
import numpy as np

import matplotlib.pyplot as plt

from networkx.readwrite import json_graph

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, GridSearchCV
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB, GaussianNB
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import tree

import math
import sys
import re

## Compute Number of occurrences in asm text

def number_of_occurrences(regex_text,text):
    return len(re.findall(regex_text,text))


## Compute parameters starting from raw data

def add_data_from_graph(x_data):
    graph_list = [json_graph.adjacency_graph(cfg) for cfg in x_data.cfg]
    nodes      = []
    complexity = []
    diameter   = []
    moves      = []
    arithmetic = []
    float_op   = []
    bitwise    = []
    calls      = []
    jumps      = []
    i = 0
    for graph in graph_list:
        E = graph.number_of_edges()
        N = graph.number_of_nodes()
        P = nx.number_strongly_connected_components(graph)

        nodes.append(N)
        complexity.append(E - N + P)
        
        if(nx.is_strongly_connected(graph)):
            diameter.append(nx.diameter(graph))
        else:
            diameter.append(0)
        
        moves.append(number_of_occurrences(r"[a-zA-Z0-9]*mov|lea[a-zA-Z0-9]*",str(x_data.lista_asm[i])))
        arithmetic.append(number_of_occurrences(r"[a-zA-Z0-9]*add|mul|sub[a-zA-Z0-9]*",str(x_data.lista_asm[i])))
        float_op.append(number_of_occurrences(r"[a-zA-Z0-9]*xmm[a-zA-Z0-9]*",str(x_data.lista_asm[i])))
        bitwise.append(number_of_occurrences(r"[a-zA-Z0-9]*and|or|xor[a-zA-Z0-9]*",str(x_data.lista_asm[i])))
        calls.append(number_of_occurrences(r"[a-zA-Z0-9]*call[a-zA-Z0-9]*",str(x_data.lista_asm[i])))
        jumps.append(number_of_occurrences(r"[a-zA-Z0-9]*jmp|jne|je|j[a-zA-Z0-9]*",str(x_data.lista_asm[i])))

        ### Print Progress
        i+=1
        score = (i/len(graph_list))*100
        sys.stdout.write("\r")
        sys.stdout.write("Elaborating Dataset: %0.3f%%"%score)
        sys.stdout.flush()
        ##################
    print()
    x_data['nodes']=nodes
    x_data['complexity']=complexity
    x_data['diameter']=diameter
    x_data['moves']=moves
    x_data['arithmetic']=arithmetic
    x_data['float_op']=float_op
    x_data['bitwise']=bitwise
    x_data['calls']=calls
    x_data['jumps']=jumps

## Train and test model with first approach

def train_and_test_model(algorithm=tree.DecisionTreeClassifier(),verbose=True):
    x_data=pd.read_json('./dataset/noduplicatedataset.json',lines=True)
    
    add_data_from_graph(x_data)
    feature_cols = ['nodes','complexity','diameter',
    'moves','arithmetic','float_op', 'bitwise','calls','jumps']
    X_all =  x_data.loc[:,feature_cols]
    Y_all =  x_data.semantic
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
    return clf,score

## Evaluate myster set with the trained model

def evaluate_mystery_set(clf):
    x_data=pd.read_json('./dataset/blindtest.json',lines=True)
    add_data_from_graph(x_data)
    feature_cols = ['nodes','complexity','diameter',
    'moves','arithmetic','float_op', 'bitwise','calls','jumps']
    X_test =  x_data.loc[:,feature_cols]
    y = clf.predict(X_test)
    return y
