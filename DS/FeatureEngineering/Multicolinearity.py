import numpy as np
import pandas as pd
from sklearn import datasets
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + str(X.iloc[:, variables].columns[maxloc]) + '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

# Load data
iris = datasets.load_iris()
X = iris.data
X = pd.DataFrame(X)
X = X.apply(pd.to_numeric)
print(X.head(10))
calculate_vif_(X)
