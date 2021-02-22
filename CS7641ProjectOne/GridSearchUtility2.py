from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import pandas as pd

baseball_data = pd.read_csv('PlayersTraining.csv', header=0)

y = baseball_data.iloc[:, 10]
X = baseball_data.iloc[:, :10]

mlp = SVC(kernel='poly')

parameter_space = {
    # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    'gamma': [0.001, 0.01, 0.1, 1, 10]
}

clf = RandomizedSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X, y)

print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))