from sklearn.model_selection import validation_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd

baseball_data = pd.read_csv('PlayersTraining.csv', header=0)

y = baseball_data.iloc[:, 10]
X = baseball_data.iloc[:, :10]
#
# param_range = ['linear', 'poly', 'rbf', 'sigmoid']
# train_scores, test_scores = validation_curve(
#     SVC(), X, y, param_name="kernel",
#     param_range=param_range, scoring="accuracy", n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# plt.title("Validation Curve with Support Vector Machines \n Kernels")
# plt.xlabel("Kernel")
# plt.ylabel("Accuracy")
# plt.ylim(0.70, 1)
# # plt.xlim(1, 10)
# lw = 2
# plt.plot(param_range, train_scores_mean, label="Training score",
#          color="darkorange", lw=lw)
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.2,
#                  color="darkorange", lw=lw)
# plt.plot(param_range, test_scores_mean, label="Cross-validation score",
#          color="navy", lw=lw)
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.2,
#                  color="navy", lw=lw)
# plt.legend(loc="best")
#
# # plt.savefig('SVMValCurve-Kernel.png')
# plt.show()


# param_range = [0.001, 0.01]
param_range = [0.001, 0.010, 0.020, 0.030, 0.040, 0.050]
train_scores, test_scores = validation_curve(
    SVC(kernel='poly'), X, y, param_name="gamma",
    param_range=param_range, scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with Support Vector Machines \n Gamma Value")
plt.xlabel("Gamma Value")
plt.ylabel("Accuracy")
plt.ylim(0.00, 1)
# plt.xlim(1, 10)
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

# plt.savefig('SVMValCurve-gamma.png')
plt.show()

# # param_range = [0.001, 0.01]
# param_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2]
# train_scores, test_scores = validation_curve(
#     SVC(), X, y, param_name="C",
#     param_range=param_range, scoring="accuracy", n_jobs=1)
# train_scores_mean = np.mean(train_scores, axis=1)
# train_scores_std = np.std(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)
# test_scores_std = np.std(test_scores, axis=1)
#
# plt.title("Validation Curve with Support Vector Machines \n Regularization")
# plt.xlabel("Regularization Value")
# plt.ylabel("Accuracy")
# plt.ylim(0.70, 1)
# # plt.xlim(1, 10)
# lw = 2
# plt.plot(param_range, train_scores_mean, label="Training score",
#          color="darkorange", lw=lw)
# plt.fill_between(param_range, train_scores_mean - train_scores_std,
#                  train_scores_mean + train_scores_std, alpha=0.2,
#                  color="darkorange", lw=lw)
# plt.plot(param_range, test_scores_mean, label="Cross-validation score",
#          color="navy", lw=lw)
# plt.fill_between(param_range, test_scores_mean - test_scores_std,
#                  test_scores_mean + test_scores_std, alpha=0.2,
#                  color="navy", lw=lw)
# plt.legend(loc="best")
#
# # plt.savefig('SVMValCurve-regularization.png')
# plt.show()
