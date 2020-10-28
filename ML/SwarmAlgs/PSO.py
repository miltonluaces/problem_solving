from pyswarm import pso

# Function to optimize
def func1(x):
    x1 = x[0]
    x2 = x[1]
    return x1**4 - 2*x2*x1**2 + x2**2 + x1**2 - 2*x1 + 5

# Constraints function
def constraints(x):
    x1 = x[0]
    x2 = x[1]
    return [-(x1 + 0.25)**2 + 0.75*x2]

# Bounds 
lwrBound = [-3, -1]
uprBound = [2, 6]

xOpt, fOpt = pso(func1, lwrBound, uprBound, f_ieqcons=constraints)
print(xOpt, fOpt)

