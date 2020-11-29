import first_approach  as fa
import second_approach as sa


from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


from time import time
import matplotlib.pyplot as plt
import numpy as np

## Return the evental classifying differences between 
## two result arrays

def compare_lists(l1,l2):
    l = []
    for i in range(len(l1)):
        if(l1[i] != l2[i]):
            l.append((i,l1[i],l2[i]))
    return l


## Measures the performances of a certain approach with the selected model

def benchmark(clf,Approach=1,vect_type="hash"):
    print('_' * 80)
    print("Training: ")
    print(clf)
    clf_trained = None
    score = 0
    t0 = time()
    if Approach  == 1:
        clf_trained,score,vectorizer = fa.train_and_test_model(clf,vect_type="hash")
    elif Approach == 2:
        clf_trained,score = sa.train_and_test_model(clf)
    else:
        raise RuntimeError("Only two approaches known : Bucket of Words[1] or Manual Graph Analysis[2]")
    train_time = time() - t0
    print("train and test time: %0.3fs" % train_time)

    t0 = time()
    if Approach == 1:
        fa.evaluate_mystery_set(clf_trained,vect=vectorizer)
    elif Approach == 2:
        sa.evaluate_mystery_set(clf_trained)
    
    test_time = time() - t0
    print("test with mystery dataset time:  %0.3fs" % test_time)

    print("accuracy:   %0.3f" % score)
    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


## Generates the output on a .txt file of the test on blind-dataset by using
## two approaches with the selected models

def output_to_file(clf1=None,v_type="hash",clf2=None,output=True,diff=True):
    
    print()
    print("##################################")
    print("######## First Approach   ########")
    print("##################################")
    print()

    if clf1 is None:
        clf_trained1,_,vect = fa.train_and_test_model(vect_type=v_type,verbose=True)
    else:
        clf_trained1,_,vect = fa.train_and_test_model(clf1,vect_type=v_type,verbose=True)
    print()
    print("##################################")
    print("######## Second Approach  ########")
    print("##################################")
    print()

    if clf2 is None:
        clf_trained2,_ = sa.train_and_test_model(verbose=True)
    else:
        clf_trained2,_ = sa.train_and_test_model(clf2,verbose=True)
    y1=fa.evaluate_mystery_set(clf_trained1,vect=vect)
    y2=sa.evaluate_mystery_set(clf_trained2)
    if diff:
        assert len(y1) == len(y2)
        diff = compare_lists(y1,y2)
        print("Over "+str(len(y1))+" occurrences:")
        print("Number     of Differences " + str(len(diff)))
        perc = (len(diff)/len(y1)) * 100
        print("Percentage of Differences %0.3f%%" % perc )
    if output:
        title1 = "results1_"+v_type+".txt"
        f = open(title1, "w")
        for element in y1:
            f.write(element+'\n')
        f.close()
        f = open("results2.txt", "w")
        for element in y2:
            f.write(element+'\n')
        f.close()

## Measures the performances of a certain approach with different models

def test_approach(Appr=1,vect_t="none"):
    results = []
    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="auto"), "Ridge Classifier"),
            (PassiveAggressiveClassifier(max_iter=50),
            "Passive-Aggressive"),
            (KNeighborsClassifier(n_neighbors=10), "kNN"),
            (RandomForestClassifier(), "Random forest"),
            (LinearSVC(),"Linear SVC"),
            (DecisionTreeClassifier(),"Desition Tree"),
            (MultinomialNB(alpha=.01),"MultinomialNB"),
            (BernoulliNB(alpha=.01),"BernoulliNB"),
            (ComplementNB(alpha=.1),"ComplementNB")
            ):
        # Naive Byes won't work with negative values
        if name in ["BernoulliNB","MultinomialNB","ComplementNB"] and vect_t in ["hash","tfid","count"]:
            continue
        print('=' * 80)
        print(name)
        print()
        print("Approach: "+str(Appr)+", Vectorizer Transform: "+vect_t)
        results.append(benchmark(clf,Approach=Appr,vect_type=vect_t))
    print('=' * 80)
    indices = np.arange(len(results))
    return results,indices

## Plot the results of the benchmark

def plot_results(results,indices,title=None):
    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    if title is None:
        plt.title("Score")
    else:
        plt.title(title)
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2, label="training time",
            color='c')
    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)
    plt.show()
def best_in_results(results):
    max_ = 0
    clf = ""
    for clf_descr,accuracy, _, _ in results:
        if accuracy > max_:
            max_ = accuracy
            clf  = clf_descr
    return max_,clf 

#output_to_file(clf1=PassiveAggressiveClassifier(),v_type="tfid",clf2=RandomForestClassifier())


results1h,indices1h=test_approach(Appr=1,vect_t="hash")
max_1h,clf1h=best_in_results(results1h)

results1c,indices1c=test_approach(Appr=1,vect_t="count")
max_1c,clf1c=best_in_results(results1c)

results1r,indices1r=test_approach(Appr=1,vect_t="tfid")
max_1r,clf1r=best_in_results(results1r)


results2,indices2=test_approach(Appr=2,vect_t="none")
max_2,clf2=best_in_results(results2)

print("Best in categories:")
print()
print(max_1h,clf1h)
print(max_1c,clf1c)
print(max_1r,clf1r)

print(max_2,clf2)

plot_results(results1h,indices1h,title="1st Approach with Hash Vectorizer")
plot_results(results1c,indices1c,title="1st Approach with Count Vectorizer")
plot_results(results1r,indices1r,title="1st Approach with Tfid Vectorizer")

plot_results(results2,indices2,title="2nd Approach")
#Best in categories:
#0.9846762234305487 PassiveAggressiveClassifier
#0.9817103311913 PassiveAggressiveClassifier
#0.9831932773109243 PassiveAggressiveClassifier
#0.9688581314878892 RandomForestClassifier
