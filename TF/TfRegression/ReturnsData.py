
import pandas as pd
import numpy as np

def read_goog_sp500_data():
    googFile = 'data/Goog.csv'
    spFile = 'data/Sp500.csv'

    goog = pd.read_csv(googFile, sep=",", usecols=[0,5], names=['Date', 'Goog'], header=0)
    sp = pd.read_csv(spFile, sep=",", usecols=[0,5], names=['Date', 'SP500'], header=0)

    goog['SP500'] = sp['SP500']
    goog['Date'] = pd.to_datetime(goog['Date'], format='%Y-%m-%d')
    goog = goog.sort_values(['Date'], ascending=[True])

    returns = goog[[key for key in dict(goog.dtypes) if dict(goog.dtypes)[key] in ['float64','int64']]].pct_change()

    #Filter out first row (with no value for returns)
    xData = np.array(returns['SP500'])[1:]
    yData = np.array(returns['Goog'])[1:]

    return (xData, yData)