import featuretools as ft
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# Data
train = pd.read_csv("Train_UWu5bXk.csv")
test = pd.read_csv("Test_u94Q5KV.csv")
 

# Data Preparation

# Saving identifiers
testItemId = test['Item_Identifier']
testOutletId = test['Outlet_Identifier']
sales = train['Item_Outlet_Sales']
train.drop(['Item_Outlet_Sales'], axis=1, inplace=True)

combi = train.append(test, ignore_index=True)
combi.isnull().sum()

# Missing data Imputation
combi['Item_Weight'].fillna(combi['Item_Weight'].mean(), inplace = True)
combi['Outlet_Size'].fillna("missing", inplace = True)

# Data Preprocessing
combi['Item_Fat_Content'].value_counts()

# Category replacement
fat_content_dict = {'Low Fat':0, 'Regular':1, 'LF':0, 'reg':1, 'low fat':0}
combi['Item_Fat_Content'] = combi['Item_Fat_Content'].replace(fat_content_dict, regex=True)

# Feature Engineering 
combi['id'] = combi['Item_Identifier'] + combi['Outlet_Identifier']
combi.drop(['Item_Identifier'], axis=1, inplace=True)

# Creating entity set
es = ft.EntitySet(id = 'sales')

# Add a dataframe 
es.entity_from_dataframe(entity_id = 'bigmart', dataframe = combi, index = 'id')
es.normalize_entity(base_entity_id='bigmart', new_entity_id='outlet', index = 'Outlet_Identifier', 
additional_variables = ['Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])
print(es)

# DFS
featMat, featNames = ft.dfs(entityset=es, target_entity = 'bigmart', max_depth = 2, verbose = 1, n_jobs = 3)
print(featMat.columns); print(featMat.head())
featMat = featMat.reindex(index=combi['id'])
featMat = featMat.reset_index()

# Model (catboost) 
catFeats = np.where(featMat.dtypes == 'object')[0]
for i in catFeats:
    featMat.iloc[:,i] = featMat.iloc[:,i].astype('str')

# Train/test split
featMat.drop(['id'], axis=1, inplace=True)
train = featMat[:8523]
test = featMat[8523:]

# Remove uneccesary variables
train.drop(['Outlet_Identifier'], axis=1, inplace=True)
test.drop(['Outlet_Identifier'], axis=1, inplace=True)

# Identify categorical features
catFeats = np.where(train.dtypes == 'object')[0]

# Split train into train/valid
xTrain, xValid, yTrain, yValid = train_test_split(train, sales, test_size=0.25, random_state=11)

model = CatBoostRegressor(iterations=100, learning_rate=0.3, depth=6, eval_metric='RMSE', random_seed=7)
model.fit(xTrain, yTrain, cat_features=catFeats, use_best_model=True)
acc = model.score(xValid, yValid); print(acc)





