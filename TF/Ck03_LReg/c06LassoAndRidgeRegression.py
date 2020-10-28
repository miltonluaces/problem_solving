# Lasso and Ridge Regression
#---------------------------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

# Load the data: iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
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

# Lasso Loss function = L2_Loss + heavysideStep, where  heavysideStep ~ 0 if A < constant, otherwise ~ 99
lassoParam = tf.constant(0.9)
heavysideStep = tf.truediv(1., tf.add(1., tf.exp(tf.multiply(-100., tf.subtract(A, lassoParam)))))
regularizParam = tf.multiply(heavysideStep, 99.)
lLoss = tf.add(tf.reduce_mean(tf.square(Ytarget - res)), regularizParam)

# Ridge loss function = L2_loss + L2 norm of slope
ridgeParam = tf.constant(1.)
ridgeLoss = tf.reduce_mean(tf.square(A))
rLoss = tf.expand_dims(tf.add(tf.reduce_mean(tf.square(Ytarget - res)), tf.multiply(ridgeParam, ridgeLoss)), 0)

# Declare optimizer
opt = tf.train.GradientDescentOptimizer(0.001)
step = opt.minimize(rLoss)

# Initialize variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
lossVec = []
for i in range(1500):
    index = np.random.choice(len(X), size=bSize)
    x = np.transpose([X[index]])
    y = np.transpose([Y[index]])
    sess.run(step, feed_dict={Xdata: x, Ytarget: y})
    ls = sess.run(rLoss, feed_dict={Xdata: x, Ytarget: y})
    lossVec.append(ls[0])
    if (i+1)%300 == 0:
        print('Step : ' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(ls))

# Get the optimal coefficients
[slope] = sess.run(A)
[intercept] = sess.run(b)

# Get best fit line
line = []
for i in X:
  line.append(slope*i + intercept)

# Plot the result
plt.plot(X, Y, 'o', label='Data Points')
plt.plot(X, line, 'r-', label='Best fit line', linewidth=3)
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
