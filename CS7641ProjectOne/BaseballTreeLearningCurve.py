from sklearn import tree
from sklearn.model_selection import learning_curve
import graphviz
import pandas as pd

baseball_data = pd.read_csv('PlayersTraining.csv', header=0)

y = baseball_data.iloc[:, 10]
X = baseball_data.iloc[:, :10]

train_sizes, train_scores, validation_scores = learning_curve(
    estimator=tree.DecisionTreeClassifier(random_state=0, max_depth=4),
    X = baseball_data.iloc[:, :10],
    y = baseball_data.iloc[:, 10],
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    cv = 10)

print('Training scores:\n\n', train_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', validation_scores)