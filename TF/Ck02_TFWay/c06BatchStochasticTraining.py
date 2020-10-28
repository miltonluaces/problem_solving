# Batch and Stochastic Training
#----------------------------------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Stochastic Training:

# Create data
X = np.random.normal(1, 0.1, 100)
Y = np.repeat(10., 100)
Xdata = tf.placeholder(shape=[1], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[1], dtype=tf.float32)

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(shape=[1]))

# Add operation to graph
res = tf.multiply(Xdata, A)

# Add L2 loss operation to graph
loss = tf.square(res - Ytarget)

# Initialize 
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Create Optimizer
opt = tf.train.GradientDescentOptimizer(0.02)
step = opt.minimize(loss)

# Initialize 
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Run
loss = []
for i in range(100):
    index = np.random.choice(100)
    x = [X[index]]
    y = [Y[index]]
    sess.run(step, feed_dict={Xdata: x, Ytarget: y})
    if (i+1)%5 == 0:
        print('Step : ' + str(i+1) + ' A = ' + str(sess.run(A)))
        ls = sess.run(loss, feed_dict={Xdata: x, Ytarget: y})
        print('Loss = ' + str(ls))
        loss.append(ls)
        

# Batch Training:

# Declare batch size
bSize = 20

# Create data
X = np.random.normal(1, 0.1, 100)
Y = np.repeat(10., 100)
Xdata = tf.placeholder(shape=[None, 1], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(shape=[1,1]))

# Add operation to graph
res = tf.matmul(Xdata, A)

# Add L2 loss operation to graph
loss = tf.reduce_mean(tf.square(res - Ytarget))

# Create Optimizer
opt = tf.train.GradientDescentOptimizer(0.02)
step = opt.minimize(loss)

# Initialize 
tf.reset_default_graph()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Run 
loss = []
for i in range(100):
    index = np.random.choice(100, size=bSize)
    x = np.transpose([X[index]])
    y = np.transpose([Y[index]])
    sess.run(step, feed_dict={Xdata: x, Ytarget: y})
    if (i+1)%5 == 0:
        print('Step : ' + str(i+1) + ' A = ' + str(sess.run(A)))
        ls = sess.run(loss, feed_dict={Xdata: x, Ytarget: y})
        print('Loss = ' + str(ls))
        loss.append(ls)

# Plot 
plt.plot(range(0, 100, 5), loss, 'b-', label='Stochastic Loss')
plt.plot(range(0, 100, 5), loss, 'r--', label='Batch Loss, size=20')
plt.legend(loc='upper right', prop={'size': 11})
plt.show()