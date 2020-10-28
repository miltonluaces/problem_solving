# Linear Regression: Solve LR by Inverse Matrix
#----------------------------------------------
#
# Given Ax=b, solving for x: x = (t(A) * A)^-1 * t(A) * b


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import Functions as fn


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

# Matrix inverse solution
tA_A = tf.matmul(tf.transpose(A), A)
tA_A_inv = tf.matrix_inverse(tA_A)
prod = tf.matmul(tA_A_inv, tf.transpose(A))
sol = tf.matmul(prod, b)

# Calculate
sess = tf.Session()
solEval = sess.run(sol)

# Extract coefficients
slope = solEval[0][0]
intercept = solEval[1][0]

print('slope: ' + str(slope))
print('intercept: ' + str(intercept))

# Get best fit line
line = []
for i in X:
  line.append(slope*i + intercept)

# Plot the results
plt.plot(X, Y, 'o', label='Data')
plt.plot(X, line, 'r-', label='fit line', linewidth=3)
plt.legend(loc='upper left')
plt.show()