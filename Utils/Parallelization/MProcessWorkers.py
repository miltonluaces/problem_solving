from Helpers.testing import *
from multiprocessing import Process,Manager
from TSAnalysis.TsIntegrals import OptimIrpe
import time
import Plotting.TsPlots as pp

def f(retDict):
    points = OptimIrpe(retDict['ts'], retDict['decRate'])
    retDict['points'] = points

if __name__ == '__main__':
    
    # Data
    dfToy = pd.read_csv(dataPath + 'toy.csv')
    ts1 = dfToy.iloc[:,1].values

    #points = OptimIrpe(ts=ts1, decRate=0.999, trace=True)
    #pp.PlotTsPoints2(ts1, points)
    #pp.PlotTsPoints2Interp(ts1, points)

    nProc=23
  
    # Sequential
    t1=time.time()
    for t in range(nProc):
        points = OptimIrpe(ts=ts1, decRate=0.999, trace=False)

    t2=time.time()
    deltaSeq = t2-t1
    #print(points)
 
    # Parallel processing
    t1=time.time()
  
    mgr = Manager()
    retDict = mgr.dict()
    retDict['ts'] = ts1
    retDict['decRate'] = 0.999
    retDict['points'] = []

    #print(retDict['points'])
    Procs = []
    for i in range(nProc):
        p = Process(target=f, args=(retDict,))
        Procs.append(p)

    for i in range(nProc):
        Procs[i].start()

    for i in range(nProc):
        Procs[i].join()

    print(retDict['points'])

    t2=time.time()
    deltaMP = t2-t1
    print("Sequential : ", '{0:.2f}'.format(deltaSeq))
    print("Multiprocessing : ", '{0:.2f}'.format(deltaMP))
    print("Improvement : ", '{0:.2f}'.format(((1 - deltaMP/deltaSeq))*100))





#def f(d):
#    d['a'] = 1

#if __name__ == '__main__':
#    manager = Manager()

#    d = manager.dict()
#    d['a'] = 0


#    print(d)

#    p = Process(target=f, args=(d,))
#    p.start()
#    p.join()

#    print(d)


