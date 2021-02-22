"""
========================
Plotting Learning Curves
========================

On the left side the learning curve of a D-Tree classifier is shown for
the Baseball HOF dataset. Note that the training score and the cross-validation score
are both not very good at the end. However, the shape of the curve can be found
in more complex datasets very often: the training score is very high at the
beginning and decreases and the cross-validation score is very low at the
beginning and increases. On the right side we see the learning curve of a
K Nearest Neighbors model. We can see clearly that the training score is still around
the maximum and the validation score could be increased with more training
samples.
"""

# print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import ShuffleSplit
import graphviz
import time

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.05, 1.0, 20)):
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
    plt.xlabel("No. of Training Examples")
    plt.ylabel("Accuracy")
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
             label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

pima_data = pd.read_csv('PimaTraining.csv', header=0)

y = pima_data.iloc[:, 8]
X = pima_data.iloc[:, :8]

tic = time.perf_counter()
title = "Learning Curve (Decision Tree)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# cv = 10
estimator = DecisionTreeClassifier(random_state=0, max_depth=3,
                                   splitter="best", ccp_alpha=0.015)
# print("Now processing D-Tree")
plot_learning_curve(estimator, title, X, y, ylim=(0.50, 1.10), cv=cv, n_jobs=4)
toc = time.perf_counter()
print(f"Pima D-Tree Time: {toc - tic:0.4f} seconds")

tic = time.perf_counter()
title = "Learning Curve (K-Nearest Neighbor)"
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
cv = 10
estimator = KNeighborsClassifier(n_neighbors=20)
# print("Now processing KNN")
plot_learning_curve(estimator, title, X, y, ylim=(0.60, 1.00), cv=cv, n_jobs=4)
toc = time.perf_counter()
print(f"Pima KNN Time: {toc - tic:0.4f} seconds")

tic = time.perf_counter()
title = "Learning Curve (Neural Network)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# cv = 10
estimator = MLPClassifier(solver='sgd', max_iter=1000,
                          learning_rate_init=0.0050,
                          hidden_layer_sizes=(10,10,10,10,10))
# print("Now processing Neural Network")
plot_learning_curve(estimator, title, X, y, ylim=(0.50, 1.10), cv=cv, n_jobs=4)
toc = time.perf_counter()
print(f"Pima N-Net Time: {toc - tic:0.4f} seconds")

tic = time.perf_counter()
title = "Learning Curve (Support Vector Machines)"
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
cv = 10
# estimator = SVC()
estimator = SVC(kernel='linear', C=0.5)
# print("Now processing SVM")
plot_learning_curve(estimator, title, X, y, ylim=(0.50, 1.10), cv=cv, n_jobs=4)
toc = time.perf_counter()
print(f"Pima SVM Time: {toc - tic:0.4f} seconds")

tic = time.perf_counter()
title = "Learning Curve (Boosting)"
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
cv = 10
estimator = AdaBoostClassifier(n_estimators=10, learning_rate=0.007,
                               random_state=0)
# print("Now processing Boosting")
plot_learning_curve(estimator, title, X, y, ylim=(0.50, 1.10), cv=cv, n_jobs=4)
toc = time.perf_counter()
print(f"Pima Boosting Time: {toc - tic:0.4f} seconds")

plt.show()