from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
import pandas as pd

baseball_data = pd.read_csv('HittersTraining.csv', header=0)

y = baseball_data.iloc[:, 7]
X = baseball_data.iloc[:, :7]

clf = tree.DecisionTreeClassifier(random_state=0, max_depth=4)
clf = clf.fit(X, y)
print(tree.plot_tree(clf))

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("baseball_data1")

feature_names = baseball_data.columns[:7]

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=feature_names,
                                class_names=['C','I','NC'],
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("baseball_training")
