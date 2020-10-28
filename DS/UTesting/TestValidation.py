from Utils.Admin.testing import *
import ModelValidation.CrossValidation as vc
import numpy as np
import pandas as pd

class TestValidation(unittest.TestCase):

    def test10_CrossValidation(self):
        dataset = pd.read_csv(dataPath + 'wineQualityReds.csv', sep=';')  
        print(dataset.head())

        X = dataset.iloc[:, 0:11].values  
        Y = dataset.iloc[:, 11].values   
        accurs = vc.CrossVal(X=X, Y=Y, testSize=0.3, nEstims=300, kFold=5)

        print(accurs)
        print(accurs.mean())  
        print(accurs.std())  


if __name__ == '__main__':
    unittest.main()


