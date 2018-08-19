#!/usr/bin/python3
#CS7641 HW1 by Tian Mi

import csv
import numpy as np
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
import sklearn.tree as tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

#################################################
#wine quality data set

data = pd.read_csv('winequality-data.csv')
X = data.iloc[:,:11]
y = data.iloc[:,11]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
features = list(X_train.columns.values)

#Decision Tree classifier
#decision tree learning curve of tree depth 5
list1=[]
list2=[]
for i in range(1,95):
    clf = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=5)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1-i/100)
    clf = clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
plt.plot(range(len(list2)),list2)
plt.plot(range(len(list1)),list1)
plt.show()

#decision tree learning curve of different function of split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
list1=[]
list2=[]
for depth in range(3,40):
    clf = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=depth)
    clf = clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_test, test_predict))
    
    clf = DecisionTreeClassifier(random_state=0, criterion='entropy', max_depth=depth)
    clf = clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list2.append(accuracy_score(y_test, test_predict))
plt.plot(range(len(list2)),list2)
plt.plot(range(len(list1)),list1)
plt.show()

#choose tree depth of 5 as optimal solution
clf = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=5)
clf = clf.fit(X_train, y_train)
test_predict = clf.predict(X_test)
print("The prediction accuracy of decision tree is " + str(accuracy_score(y_test, test_predict)))

#visualization of decision tree
#dot_data = tree.export_graphviz(clf, out_file=None, 
#                         feature_names=features,  
#                         class_names=list(map(str, set(y))),  
#                         filled=True, rounded=True,  
#                         special_characters=True)
#graph = graphviz.Source(dot_data)
#graph

#Neural network classifier learning curve
list1=[]
list2=[]
for i in range(1,95):
    clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(20, 5), random_state=0, activation='logistic')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
plt.plot(range(len(list2)),list2)
plt.plot(range(len(list1)),list1)
plt.show()

#Boosted DT classifier
list1=[]
list2=[]
for i in range(1,95):
    clf = clf = AdaBoostClassifier(n_estimators=100)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
plt.plot(range(len(list2)),list2)
plt.plot(range(len(list1)),list1)
plt.show()

#SVM classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
for kernel in ('linear', 'rbf'):
    clf = svm.SVC(kernel=kernel, gamma=2)
    clf = clf.fit(X_train, y_train)
    test_predict = clf.predict(X_test)
    print(accuracy_score(y_test, test_predict))

#Choose RBF as the preferred kernel function
clf = svm.SVC(kernel="rbf", gamma=2)
clf = clf.fit(X_train, y_train)
test_predict = clf.predict(X_test)
print("The prediction accuracy of SVM with RBF kernel is " + str(accuracy_score(y_test, test_predict)))

#SVM learning curve with RBF kernel
list1=[]
list2=[]
for i in range(1,95):
    clf = svm.SVC(kernel="rbf", gamma=2)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
plt.ylim(ymin=0,ymax=1.1)
plt.plot(range(len(list2)),list2)
plt.plot(range(len(list1)),list1)
plt.show()

#KNN classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
KNN_list=[]
list2=[]
for K in range(1,50):
    clf = KNeighborsClassifier(K, weights="distance")
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    clf = clf.fit(X_train, y_train)
    test_predict = clf.predict(X_test)
    KNN_list.append(accuracy_score(y_test, test_predict))
    list2.append(sum(scores)/len(scores))
plt.plot(range(len(KNN_list)),KNN_list)
plt.plot(range(len(list2)),list2)
plt.show()

#choose 13 as the optimal K
clf = KNeighborsClassifier(13, weights="distance")
clf = clf.fit(X_train, y_train)
test_predict = clf.predict(X_test)
print("The prediction accuracy of KNN classifier with k=13 is " + str(accuracy_score(y_test, test_predict)))

#learning curve of KNN classifier with K=13
list1=[]
list2=[]
for i in range(1,95):
    clf = KNeighborsClassifier(13, weights="distance")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1-i/100)
    clf.fit(X_train, y_train)
    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)
    list1.append(accuracy_score(y_train, train_predict))
    list2.append(accuracy_score(y_test, test_predict))
plt.plot(range(len(list2)),list2)
plt.plot(range(len(list1)),list1)
plt.show()

#########################################
#learning curve function from sklearn tutorial

from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

###############################################
#Titanic survival data set
data = pd.read_csv('titanic_train.csv')
X = data.iloc[:,2:]
y = data.iloc[:,1]
features = list(X.columns.values)

#Decision Tree classifier
#decision tree learning curve of tree depth 3
clf = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=3)
plot_learning_curve(clf, "Decision Tree with max depth 3", X, y, ylim=[0,1])

clf = clf.fit(X, y)
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=features,  
                         class_names=list(map(str, set(y))),  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph

#Neural network classifier
clf = MLPClassifier(hidden_layer_sizes=(5), random_state=0, solver="lbfgs")
plot_learning_curve(clf, "MLP with hidden layers as (5)", X, y, ylim=[0,1])

#Boosted DT classifier
clf = AdaBoostClassifier(n_estimators=500)
plot_learning_curve(clf, "Adaboost with n_estimators 500", X, y, ylim=[0,1])

clf = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.001, max_depth=3, random_state=0, max_leaf_nodes=5)
plot_learning_curve(clf, "Gradient Boosting with n_estimators 1000", X, y, ylim=[0,1])

#SVM classifier
from sklearn.model_selection import GridSearchCV
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100],
                     'C': [0.01, 0.1, 1, 10, 100, 1000]} ]
clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring='accuracy')
clf.fit(X, y)
print(clf.best_params_)

clf = svm.SVC(C=10, kernel="rbf", gamma=0.01)
plot_learning_curve(clf, "SVM with RBF kernel, gamma=0.01", X, y, ylim=[0,1])

clf = svm.SVC(C=100, kernel="linear", gamma=0.0001)
plot_learning_curve(clf, "SVM with linear kernel, gamma=0.0001", X, y, ylim=[0,1])

#KNN classifier
clf = KNeighborsClassifier(1, weights="distance", p=2)
plot_learning_curve(clf, "K nearest neighbors with K=1", X, y, ylim=[0,1])

clf = KNeighborsClassifier(5, weights="distance", p=2)
plot_learning_curve(clf, "K nearest neighbors with K=5", X, y, ylim=[0,1])

clf = KNeighborsClassifier(10, weights="distance", p=2)
plot_learning_curve(clf, "K nearest neighbors with K=10", X, y, ylim=[0,1])

clf = KNeighborsClassifier(10, weights="uniform", p=2)
plot_learning_curve(clf, "K nearest neighbors with K=10, uniform weights", X, y, ylim=[0,1])
