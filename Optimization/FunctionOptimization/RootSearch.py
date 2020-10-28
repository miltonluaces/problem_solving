import numpy as np
from scipy import optimize

# Function definition
def f1(z, *params):
     x, y = z
     a, b, c, d, e, f, g, h, i, j, k, l, scale = params
     return (a * x**2 + b * x * y + c * y**2 + d*x + e*y + f)

def f2(z, *params):
     x, y = z
     a, b, c, d, e, f, g, h, i, j, k, l, scale = params
     return (-g*np.exp(-((x-h)**2 + (y-i)**2) / scale))

def f3(z, *params):
     x, y = z
     a, b, c, d, e, f, g, h, i, j, k, l, scale = params
     return (-j*np.exp(-((x-k)**2 + (y-l)**2) / scale))

def f(z, *params):
     return f1(z, *params) + f2(z, *params) + f3(z, *params)

# Testing
rranges = (slice(-4, 4, 0.25), slice(-4, 4, 0.25))
params = (2, 3, 7, 8, 9, 10, 44, -1, 2, 26, 1, -2, 0.5)
resbrute = optimize.brute(f, rranges, args=params, full_output=True, finish=optimize.fmin)
print('global minimum : ', resbrute[0]) 
print('function value at global minimum :', resbrute[1])
