# minimize:     c^T * x
# subject to:   A_ub * x <= b_ub and A_eq * x == b_eq
# c : array of Coefficients of the linear objective function to be minimized
# A_ub : 2-D array which, when matrix-multiplied by x, gives the values of the upper-bound inequality constraints at x
# b_ub : 1-D array of values representing the upper-bound of each inequality constraint (row) in A_ub
# A_eq : 2-D array which, when matrix-multiplied by x, gives the values of the equality constraints at x
# b_eq : 1-D array of values representing the RHS of each equality constraint (row) in A_eq

from scipy.optimize import linprog

# Load data
c = [-1, 4]
A = [[-3, 1], [1, 2]]
b = [6, 4]
x0bnds = (None, None)
x1bnds = (-3, None)

res = linprog(c, A, b, bounds=(x0bnds, x1bnds))
print(res)
