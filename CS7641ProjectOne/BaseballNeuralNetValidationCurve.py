from sklearn.model_selection import validation_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import pandas as pd

baseball_data = pd.read_csv('PlayersTraining.csv', header=0)

y = baseball_data.iloc[:, 10]
X = baseball_data.iloc[:, :10]

# param_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# train_scores, test_scores = validation_curve(
#     MLPClassifier(solver='adam', max_iter=1000), X, y, param_name="momentum",
#     param_range=param_range,
#     scoring="accuracy", n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# plt.title("Validation Curve for Neural Network \n Momentum")
# plt.xlabel("Momentum")
# plt.ylabel("Accuracy")
# plt.ylim(0.00, 1.00)
# plt.xlim(0.1, 0.9)
# lw = 2
# plt.plot(param_range, train_scores_mean, label="Training score",
#              color="darkorange", lw=lw)
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.2,
#                  color="darkorange", lw=lw)
# plt.plot(param_range, test_scores_mean, label="Cross-validation score",
#              color="navy", lw=lw)
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.2,
#                  color="navy", lw=lw)
# plt.legend(loc="best")
# plt.show()
#
# param_range = [(10), (10,10), (10,10,10),
#                            (10,10,10,10), (10,10,10,10,10),
#                            (10,10,10,10,10,10), (10,10,10,10,10,10,10),
#                            (10,10,10,10,10,10,10,10),
#                            (10,10,10,10,10,10,10,10,10),
#                            (10,10,10,10,10,10,10,10,10,10)]
# printable_range = [1,2,3,4,5,6,7,8,9,10]
# train_scores, test_scores = validation_curve(
#     MLPClassifier(solver='adam', max_iter=1000), X, y, param_name="hidden_layer_sizes",
#     param_range=param_range,
#     scoring="accuracy", n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# plt.title("Validation Curve for Neural Network \n Number of Hidden Layers")
# plt.xlabel("Number of Hidden Layers")
# plt.ylabel("Accuracy")
# plt.ylim(0.00, 1.00)
# plt.xlim(1, 10)
# lw = 2
# plt.plot(printable_range, train_scores_mean, label="Training score",
#              color="darkorange", lw=lw)
# plt.fill_between(printable_range, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.2,
#                  color="darkorange", lw=lw)
# plt.plot(printable_range, test_scores_mean, label="Cross-validation score",
#              color="navy", lw=lw)
# plt.fill_between(printable_range, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.2,
#                  color="navy", lw=lw)
# plt.legend(loc="best")
# plt.show()

param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
               21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
train_scores, test_scores = validation_curve(
    MLPClassifier(solver='adam', hidden_layer_sizes=(10,10,10,10,10,10,10,10),
                  learning_rate_init=0.0050), X, y, param_name="max_iter",
    param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Learning Curve (Neural Network) \n Max Iterations")
plt.xlabel("Max Iterations")
plt.ylabel("Accuracy")
plt.ylim(0.00, 1.00)
# plt.xlim(0.001, 0.02)
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
plt.show()

# param_range = [0.001, 0.003, 0.005, 0.007, 0.009, 0.01, 0.015, 0.02]
# train_scores, test_scores = validation_curve(
#     MLPClassifier(solver='adam', max_iter=1000), X, y, param_name="learning_rate_init",
#     param_range=param_range,
#     scoring="accuracy", n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# plt.title("Validation Curve for Neural Network \n Initial Learning Rate")
# plt.xlabel("Initial Learning Rate")
# plt.ylabel("Accuracy")
# plt.ylim(0.00, 1.00)
# plt.xlim(0.001, 0.02)
# lw = 2
# plt.plot(param_range, train_scores_mean, label="Training score",
#              color="darkorange", lw=lw)
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.2,
#                  color="darkorange", lw=lw)
# plt.plot(param_range, test_scores_mean, label="Cross-validation score",
#              color="navy", lw=lw)
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.2,
#                  color="navy", lw=lw)
# plt.legend(loc="best")
# plt.show()