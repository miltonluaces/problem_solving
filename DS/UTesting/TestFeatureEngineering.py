import unittest
import FeatureEngineering.FeatureImportance as fi
import FeatureEngineering.Multicolinearity as fm
import FeatureEngineering.FeatureClustering as fc
import Visual.Distributions.ParetoChart as pch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TestFeatureEngineering(unittest.TestCase):

    def test00_ParetoPlot(self):
        data = [21, 2, 10, 4, 16]
        labels = ['tom', 'betty', 'alyson', 'john', 'bob']
        pch.Pareto(data, labels,  limit=0.975)
        plt.show()

    def test10_FeatureImportance(self):
        # Load data
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        labels = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
        dataframe = pd.read_csv(url, names=labels)
        array = dataframe.values
        X = array[:,0:8]
        Y = array[:,8]
        print(X)
        print(Y)
        importances = fi.FeatureImportance(X,Y)
        pch.Pareto(importances, labels[:8],  limit=0.975)
        plt.title('Feature importance', fontsize=10)
        plt.show()

if __name__ == '__main__':
    unittest.main()


