from sklearn.model_selection import ParameterGrid
import numpy as np
#from hyperopt import hp, tpe, fmin


# Brute force
def BruteForce(func, parDict, trace=False):
    parGrid = ParameterGrid(parDict)
    bestY = 0
    bestPars = None
    for pars in parGrid:
        y = func(pars['a'], pars['b'], pars['c'], pars['x'])
        if(trace): print(pars, " = ", y)
        if(y > bestY): bestY=y; bestPars = pars
    return bestPars, bestY

# Bayesian