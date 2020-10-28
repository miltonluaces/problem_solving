import unittest
import DataMining.AnomalyDetection.OutlierFiltering as do
import DataCleansing.Inliers as di
import DataCleansing.Constraints as dc
import DataMining.AnomalyDetection.Deduping as dd
import DataCleansing.MissingImputation as dm
import DataCleansing.Wrangling as dw
import DataCleansing.WrongClassification as dg
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from numpy import nan
import recordlinkage
from recordlinkage.datasets import load_febrl1
from Visual.Misc.Tables import ShowDataFrame


class TestDataCleansing(unittest.TestCase):

    def LoadData(self):
        data = load_febrl1()
        return(data)
    
    def Indexation(self, data):
        idxr = recordlinkage.Index()
        idxr.block(left_on='given_name') # force matching in this field
        candidateLinks = idxr.index(data)
        return(candidateLinks)

    def Comparison(self, data, candidateLinks):
        cmp = recordlinkage.Compare()
        cmp.exact('given_name', 'given_name', label='given_name')
        cmp.string('surname', 'surname', method='jarowinkler', threshold=0.85, label='surname')
        cmp.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
        cmp.exact('suburb', 'suburb', label='suburb')
        cmp.exact('state', 'state', label='state')
        cmp.string('address_1', 'address_1', threshold=0.85, label='address_1')
        features = cmp.compute(candidateLinks, data)
        return(features)

    # Load data
    def test10_DataCleansing_Deduping(self):
        data = self.LoadData()
        print(data.head(10))

    # Indexation step
    def test11_DataCleansing_Deduping(self):
        data = self.LoadData()
        candidateLinks = self.Indexation(data)

    # Comparison step
    def test12_DataCleansing_Deduping(self):
        data = self.LoadData()
        candidateLinks = self.Indexation(data)
        features = self.Comparison(data, candidateLinks)

    # Classification step
    def test13_DataCleansing_Deduping(self):
        data = self.LoadData()
        candidateLinks = self.Indexation(data)
        features = self.Comparison(data, candidateLinks)
        matches1 = features[features.sum(axis=1) >= 4]
        matches2 = features[features.sum(axis=1) >= 5]
        matches3 = features[features.sum(axis=1) >= 6]
        print(matches1)
        print(len(candidateLinks))
        print(len(matches1))
        print(len(matches2))
        print(len(matches3))


    def test20_DataCleansing_MissingInputation(self):
        data = np.array([[ nan, 0,   3  ], [ 3,   7,   9  ], [ 3,   5,   2  ], [ 4,   nan, 6  ], [ 8,   8,   1  ]])
        print(data)

        # Detecting  nulls
        df = pd.DataFrame(data)
        nulls = df.isnull(); print(nulls)

        # Null imputation
        imp = Imputer(strategy='mean')
        data2 = imp.fit_transform(data)
        print(data2)



if __name__ == '__main__':
    unittest.main()

