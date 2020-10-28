import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load data
iris = datasets.load_iris()
x = iris.data[:,:2]
y = iris.target
print(x); print(y)

# Stratified split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=0.8, stratify=y)

# Check dataset is stratified (same ocurrences for each y level)
print(xTest)
print(yTest)
for i in np.unique(yTest):
    print(yTest[yTest==i].shape[0])