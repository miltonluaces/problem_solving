import pandas as pd
from multiprocessing import Pool
from TSAnalysis.TsIntegrals import OptimIrpe
import Plotting.TsPlots as pp
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':
    # Data
    dfToy = pd.read_csv('../../../data/toy.csv')
    ts1 = dfToy.iloc[:,1].values

    points = OptimIrpe(ts=ts1, decRate=0.999, trace=True)
    pp.PlotTsPoints2(ts1, points)
    pp.PlotTsPoints2Interp(ts1, points)

    nProcesses = 5
    times=[]
    for i in (range(1,nProcesses+1)):
        start = time.time()
        points=[]
        with Pool(i) as p:
            ts=ts1; decRate=0.999
            points = p.map(OptimIrpe, [ts1])
            print(points)
        end = time.time()
        delta = (end-start)/i
        print("Time taken ", id, " = {0:.5f}".format(delta))
        times.append(delta)

    print(points)
    pp.PlotTsPoints2(ts1, points)
    pp.PlotTsPoints2Interp(ts1, points)
 
    plt.plot(times)
    plt.show()

