# Linear Regression: Decomposition Method
#----------------------------------------
#
# Given Ax=b, and a Cholesky decomposition A = L*L' we can solve x with L*y = t(A)*b  and L'*x=y

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Create the data
X = np.linspace(0, 10, 100)
Y = X + np.random.normal(0, 1, 100)

# Create design matrix
XCol = np.transpose(np.matrix(X))
OnesCol = np.transpose(np.matrix(np.repeat(1, 100)))
A_ = np.column_stack((XCol, OnesCol))

# Create b matrix
b_ = np.transpose(np.matrix(Y))

# Create tensors
A = tf.constant(A_)
b = tf.constant(b_)

# Find Cholesky Decomposition
tA_A = tf.matmul(tf.transpose(A), A)
L = tf.cholesky(tA_A)

# Solve L*y = t(A)*b
tA_b = tf.matmul(tf.transpose(A), b)
sol1 = tf.matrix_solve(L, tA_b)

# Solve L'*y = sol1
sol2 = tf.matrix_solve(tf.transpose(L), sol1)

# Calculate
sess = tf.Session()
solEval = sess.run(sol2)

# Extract coefficients
slope = solEval[0][0]
intercept = solEval[1][0]

print('slope: ' + str(slope))
print('intercept: ' + str(intercept))

line = []
for i in X:
  line.append(slope*i + intercept)

# Plot the results
plt.plot(X, Y, 'o', label='Data')
plt.plot(X, line, 'r-', label='fit line', linewidth=3)
plt.legend(loc='upper left')
plt.show()