# Deming Regression
#----------------------------------
#
# This function shows how to use Tensorflow to
# solve linear Deming regression.
# y = Ax + b
#
# We will use the iris data, specifically:
#  y = Sepal Length
#  x = Petal Width

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets


# Create graph
sess = tf.Session()

# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
X = np.array([x[3] for x in iris.data])
Y = np.array([y[0] for y in iris.data])

# Declare batch size
bSize = 50

# Initialize placeholders
Xdata = tf.placeholder(shape=[None, 1], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Declare model operations
res = tf.add(tf.matmul(Xdata, A), b)

# Declare Demming loss function
demming_numerator = tf.abs(tf.sub(Ytarget, tf.add(tf.matmul(Xdata, A), b)))
demming_denominator = tf.sqrt(tf.add(tf.square(A),1))
loss = tf.reduce_mean(tf.truediv(demming_numerator, demming_denominator))

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Declare optimizer
opt = tf.train.GradientDescentOptimizer(0.1)
step = opt.minimize(loss)

# Training loop
lossVec = []
for i in range(250):
    index = np.random.choice(len(X), size=bSize)
    x = np.transpose([X[index]])
    y = np.transpose([Y[index]])
    sess.run(step, feed_dict={Xdata: x, Ytarget: y})
    ls = sess.run(loss, feed_dict={Xdata: x, Ytarget: y})
    lossVec.append(ls)
    if (i+1)%50==0:
        print('Step : ' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(ls))

# Get the optimal coefficients
[slope] = sess.run(A)
[y_intercept] = sess.run(b)

# Get best fit line
best = []
for i in X:
  best.append(slope*i+y_intercept)

# Plot the result
plt.plot(X, Y, 'o', label='Data Points')
plt.plot(X, best, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()

# Plot loss over time
plt.plot(lossVec, 'k-')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.show()
