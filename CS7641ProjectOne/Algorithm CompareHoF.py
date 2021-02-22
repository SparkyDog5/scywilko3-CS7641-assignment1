import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

# load dataset
baseball_data = pd.read_csv('PlayersTraining.csv', header=0)

y = baseball_data.iloc[:, 10]
X = baseball_data.iloc[:, :10]

# prepare configuration for cross validation test harness
seed = 5
# prepare models
models = []
models.append(('D-Tree', DecisionTreeClassifier(random_state=0, max_depth=3,
												splitter="best", ccp_alpha=0.025)))
models.append(('N-Net', MLPClassifier(solver='adam', max_iter=1000,
									  learning_rate_init=0.005,
									  hidden_layer_sizes=(10,10,10,10,10,10,10,10))))
models.append(('Boosting', AdaBoostClassifier(n_estimators=90, learning_rate=0.007)))
models.append(('SVM', SVC()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=8)))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
val_for_chart = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f %f %f (%f)" % (name, cv_results.max(), cv_results.min(),
								 cv_results.mean(), cv_results.std())
	val_for_chart.append(cv_results.mean())
	print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison\nHall of Fame Data')
ax = fig.add_subplot(111)
bp = ax.boxplot(results, showmeans=True, meanline=True)
for i, line in enumerate(bp['means']):
	x, y = line.get_xydata()[1]
	print_value = 	str(round(val_for_chart[i], 3))
	text = names[i] + '\n Mean  \n ' + print_value
	ax.annotate(text, xy=(x, y), fontsize=6)
	# plt.axhline(line_value, color='blue', linestyle='--')
plt.ylabel("Accuracy")
plt.xlabel("Algorithm")
ax.set_xticklabels(names)
plt.show()