from sklearn import tree
from sklearn import tree
import graphviz 
from sklearn.datasets import load_iris
 
# Classification
X = [[0, 0], [1, 1]]
Y = [0, 1]
dt = tree.DecisionTreeClassifier()
dt = dt.fit(X, Y)

dt.predict([[2., 2.]])
array([1])

dt.predict_proba([[2., 2.]])
array([[0., 1.]])

iris = load_iris()
dt = tree.DecisionTreeClassifier()
dt = dt.fit(iris.data, iris.target)

data = tree.export_graphviz(dt, out_file=None) 
graph = graphviz.Source(data) 
graph.render("iris") 

data = tree.export_graphviz(dt, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True,  special_characters=True)  
graph = graphviz.Source(data)  
graph 

# Regression
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
dt = tree.DecisionTreeRegressor()
dt = dt.fit(X, y)
dt.predict([[1, 1]])
array([0.5])

