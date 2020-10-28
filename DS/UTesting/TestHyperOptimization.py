import unittest
import numpy as np
import HyperOptimization.GridSearch as hg
from hyperopt import hp, tpe, fmin


class TestHyperOptimization(unittest.TestCase):

    
    def test10_GridSearch_BruteForce(self):
        def func(a, b, c, x):
            pol = np.poly1d([a, b, c]); #print(pol)
            return(pol(x))
 
        y = func(1,1,2,1); print(y)
        parDict = {'a':[1, 2, 3], 'b':[1,3,5], 'c':[2,4,6], 'x':[1,2,3,4,5]}; print(parDict)

        bestPars, bestY = hg.BruteForce(func, parDict, trace=False)
        print("Best:")
        print(bestPars, " = ", bestY)

        print("Trace: ")
        bestPars, bestY = hg.BruteForce(func, parDict, trace=True)

    def test11_GridSearch_Bayesian1Choice(self):
        def f(space):
            a = space['a']
            x = space['x']
            pol = np.poly1d([a])
            return(pol(x))

        space = { 'a' : hp.choice('a',[1,2,3]), 'x' : hp.choice('x',[1,2,3,4,5]) }
        best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=1000)
        print(best)

    def test12_GridSearch_Bayesian2Dist(self):
        def f(space):
            x = space['x']
            y = space['y']
            return x**2 + y**2
	
        space = {'x': hp.uniform('x', -5, 5), 'y': hp.uniform('y', -5, 5) }
        best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=1000)
        print(best)
  
if __name__ == '__main__':
    unittest.main()


