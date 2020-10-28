# Init Block
#%%

#from Utils.Admin.Standard import *
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# Load data
def LoadData():
    dataset = pd.read_csv('D:/data/csv/diabetes.csv', header=None)
    data = dataset.iloc[2:]
    data = pd.DataFrame(data)
    data = data.apply(pd.to_numeric)
    return(data)

# Data Analysis
def DataAnalysis(data):
    print(data.dtypes)
    print(data.head(5))

    # Primary analisis
    print(data.describe())
    print(data.head(20))

    # Identify zeros
    print((data[[1,2,3,4,5]] == 0).sum())

    # Marking
    data[[1,2,3,4,5]] = data[[1,2,3,4,5]].replace(0, np.NaN)
    print(data.head(20))
    print(data.isnull().sum()) # NaNs by column

data = LoadData()
DataAnalysis(data)

# Strategy 1 : Remove rows with missing values
#%%

data.dropna(inplace=True)
# summarize the number of rows and columns in the dataset
print(data.shape)

# Strategy 2 : Impute Missing Values (mean)
#%%

data.fillna(data.mean(), inplace=True)
print(data.head(20))
print(data.isnull().sum()) # count NaNs by column

# Strategy 3 :  LDA imputation (to review)
#%%

# split dataset into inputs and outputs
values = data.values
X = values[:,0:8]
y = values[:,8]
# fill missing values with mean column values
imputer = SimpleImputer()
transfX = imputer.fit_transform(X)
# evaluate an LDA model on the dataset using k-fold cross validation
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=3, random_state=7)
result = cross_val_score(model, transfX, y, cv=kfold, scoring='accuracy')
print(result.mean())