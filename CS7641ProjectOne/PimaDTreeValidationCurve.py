from sklearn.model_selection import validation_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn import tree
import graphviz
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

pima_data = pd.read_csv('PimaTraining.csv', header=0)

y = pima_data.iloc[:, 8]
X = pima_data.iloc[:, :8]

param_range = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
train_scores, test_scores = validation_curve(
    tree.DecisionTreeClassifier(), X, y, param_name="max_depth",
    param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with Decision Tree \n Max Depth")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.ylim(0.70, 1.00)
plt.xlim(10, 1)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig('DTreeValCurve-MaxDepth.png')
plt.show()

param_range = [0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035]
train_scores, test_scores = validation_curve(
    tree.DecisionTreeClassifier(), X, y, param_name="ccp_alpha",
    param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with Decision Tree \n Post Pruning CCP")
plt.xlabel("alpha value")
plt.ylabel("Accuracy")
plt.ylim(0.70, 1.00)
plt.xlim(0.000, 0.035)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig('DTreeValCurve-CCPPostPruning.png')
plt.show()

param_range = ['best', 'random']
train_scores, test_scores = validation_curve(
    tree.DecisionTreeClassifier(), X, y, param_name="splitter",
    param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with Decision Tree \n Splitter")
plt.xlabel("Splitter")
plt.ylabel("Accuracy")
plt.ylim(0.00, 1.00)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig('DTreeValCurve-Splitter.png')
plt.show()