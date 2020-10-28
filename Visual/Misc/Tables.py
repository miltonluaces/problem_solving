import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import asciitable as at
import tableprint as tp


# Show Table
def ShowTable(data, colNames, rowNames=None): 
    plt.table(cellText=data, rowLabels=rowNames, colLabels=colNames, loc="upper center")
    plt.axis('off')
    plt.show()

def ShowDataFrame(df):
    colNames = np.asarray(df.columns)
    plt.table(cellText=df.values, rowLabels=None, colLabels=colNames, loc="upper center")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":

    data = np.random.randn(10,3)
    headers = ['Column A', 'Column B', 'Column C']
    tp.table(data, headers)

    # Write table 
    x = np.array([1, 2, 3])
    y = x**2
    #at.write({'x': x, 'y': y}, 'outfile.dat', names=['x', 'y'])
    print("Ok")
