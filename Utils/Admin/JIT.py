from numba import jit
import random
import time


@jit(nopython=True)
def MonteCarloPi(nsamples):
    acc = 0
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x**2 + y**2) < 1.0:
            acc += 1
    return 4.0 * acc 


for nsamples in [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]: 
    start = time.time()
    MonteCarloPi(nsamples)
    end = time.time()
    print(nsamples, " : ", '{0:.2f}'.format(end-start))