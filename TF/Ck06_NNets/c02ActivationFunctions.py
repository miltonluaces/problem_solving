# Combining Gates and Activation Functions
#----------------------------------
#
# This function shows how to implement
# various gates with activation functions
# in Tensorflow
#
# This function is an extension of the
# prior gates, but with various activation
# functions.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Start Graph Session
sess = tf.Session()
tf.set_random_seed(5)
np.random.seed(42)

bSize = 50

a1 = tf.Variable(tf.random_normal(shape=[1,1]))
b1 = tf.Variable(tf.random_uniform(shape=[1,1]))
a2 = tf.Variable(tf.random_normal(shape=[1,1]))
b2 = tf.Variable(tf.random_uniform(shape=[1,1]))
x = np.random.normal(2, 0.1, 500)
Xdata = tf.placeholder(shape=[None, 1], dtype=tf.float32)

sigmoid_activation = tf.sigmoid(tf.add(tf.matmul(Xdata, a1), b1))

relu_activation = tf.nn.relu(tf.add(tf.matmul(Xdata, a2), b2))

# Declare the loss function as the difference between
# the output and a target value, 0.75.
loss1 = tf.reduce_mean(tf.square(tf.sub(sigmoid_activation, 0.75)))
loss2 = tf.reduce_mean(tf.square(tf.sub(relu_activation, 0.75)))

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Declare optimizer
opt = tf.train.GradientDescentOptimizer(0.01)
step_sigmoid = opt.minimize(loss1)
step_relu = opt.minimize(loss2)

# Run loop across gate
print('\nOptimizing Sigmoid AND Relu Output to 0.75')
lossVec_sigmoid = []
lossVec_relu = []
activation_sigmoid = []
activation_relu = []
for i in range(750):
    rand_indices = np.random.choice(len(x), size=bSize)
    X = np.transpose([x[rand_indices]])
    sess.run(step_sigmoid, feed_dict={Xdata: X})
    sess.run(step_relu, feed_dict={Xdata: X})
    
    lossVec_sigmoid.append(sess.run(loss1, feed_dict={Xdata: X}))
    lossVec_relu.append(sess.run(loss2, feed_dict={Xdata: X}))    
    
    activation_sigmoid.append(np.mean(sess.run(sigmoid_activation, feed_dict={Xdata: X})))
    activation_relu.append(np.mean(sess.run(relu_activation, feed_dict={Xdata: X})))


# Plot the activation values
plt.plot(activation_sigmoid, 'k-', label='Sigmoid Activation')
plt.plot(activation_relu, 'r--', label='Relu Activation')
plt.ylim([0, 1.0])
plt.title('Activation Outputs')
plt.xlabel('Generation')
plt.ylabel('Outputs')
plt.legend(loc='upper right')
plt.show()

    
# Plot the loss
plt.plot(lossVec_sigmoid, 'k-', label='Sigmoid Loss')
plt.plot(lossVec_relu, 'r--', label='Relu Loss')
plt.ylim([0, 1.0])
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()