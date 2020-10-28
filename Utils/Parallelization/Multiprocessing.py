from multiprocessing import Pool
import time
import matplotlib.pyplot as plt

def IsPrime(num):
    if num <= 1: return False
    elif num <= 3: return True
    elif num%2 == 0 or num%3 == 0: return False
    i = 5
    while i*i <= num:
        if num%i == 0 or num%(i+2) == 0: return False
        i += 6
    return True

def SumPrime(num):
    sum = 0
    i = 2
    while i <= num:
        if IsPrime(i): sum += i
        i += 1
    return sum


if __name__ == '__main__':
    times=[]
    for i in (range(1,10)):
        start = time.time()
        with Pool(i) as p:
            sums = p.map(SumPrime, [100000, 200000, 300000])
        end = time.time()
        delta = (end-start)/i
        print(sums)
        print("Time taken ", id, " = {0:.5f}".format(delta))
        times.append(delta)
    plt.plot(times)
    plt.show()
        