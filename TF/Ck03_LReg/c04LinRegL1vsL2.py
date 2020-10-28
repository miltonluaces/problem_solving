# Linear Regression: L1 vs L2
#----------------------------------

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

# Declare batch size and number of iterations
bSize = 25
learning_rate = 0.4 # Will not converge with learning rate at 0.4
iterations = 50

# Initialize placeholders
Xdata = tf.placeholder(shape=[None, 1], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Declare model operations
res = tf.add(tf.matmul(Xdata, A), b)

# Declare loss functions
loss_l1 = tf.reduce_mean(tf.abs(Ytarget - res))

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Declare optimizers
opt_l1 = tf.train.GradientDescentOptimizer(learning_rate)
step_l1 = opt_l1.minimize(loss_l1)

# Training loop
lossVec_l1 = []
for i in range(iterations):
    index = np.random.choice(len(X), size=bSize)
    x = np.transpose([X[index]])
    y = np.transpose([Y[index]])
    sess.run(step_l1, feed_dict={Xdata: x, Ytarget: y})
    ls_l1 = sess.run(loss_l1, feed_dict={Xdata: x, Ytarget: y})
    lossVec_l1.append(ls_l1)
    if (i+1)%25==0:
        print('Step : ' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))


# L2 Loss
# Reinitialize graph
ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Initialize placeholders
Xdata = tf.placeholder(shape=[None, 1], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Declare model operations
res = tf.add(tf.matmul(Xdata, A), b)

# Declare loss functions
loss_l2 = tf.reduce_mean(tf.square(Ytarget - res))

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Declare optimizers
opt_l2 = tf.train.GradientDescentOptimizer(learning_rate)
step_l2 = opt_l2.minimize(loss_l2)

lossVec_l2 = []
for i in range(iterations):
    index = np.random.choice(len(X), size=bSize)
    x = np.transpose([X[index]])
    y = np.transpose([Y[index]])
    sess.run(step_l2, feed_dict={Xdata: x, Ytarget: y})
    ls_l2 = sess.run(loss_l2, feed_dict={Xdata: x, Ytarget: y})
    lossVec_l2.append(ls_l2)
    if (i+1)%25==0:
        print('Step : ' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))


# Plot loss over time
plt.plot(lossVec_l1, 'k-', label='L1 Loss')
plt.plot(lossVec_l2, 'r--', label='L2 Loss')
plt.title('L1 and L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L1 Loss')
plt.legend(loc='upper right')
plt.show()