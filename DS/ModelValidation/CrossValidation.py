import pandas as pd  
import numpy as np  
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import train_test_split  


def CrossVal(X, Y, testSize, nEstims, kFold):
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=testSize, random_state=0)  

    # Scale data
    scaler = StandardScaler()  
    featTrain = scaler.fit_transform(Xtrain)  
    featTest = scaler.transform(Xtest)  

    # Cross validation
    classifier = RandomForestClassifier(n_estimators=nEstims, random_state=0)   
    accurs = cross_val_score(estimator=classifier, X=Xtrain, y=Ytrain, cv=kFold)  
    return accurs

   
