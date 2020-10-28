# Elastic Net Regression
#-----------------------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets


# Load the data: iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
X = np.array([[x[1], x[2], x[3]] for x in iris.data])
Y = np.array([y[0] for y in iris.data])

# Declare batch size
bSize = 50

# Initialize placeholders
Xdata = tf.placeholder(shape=[None, 3], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[3,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Declare model operations
res = tf.add(tf.matmul(Xdata, A), b)

# Declare the elastic net loss function
elasticParam1 = tf.constant(1.)
elasticParam2 = tf.constant(1.)
l1ALoss = tf.reduce_mean(tf.abs(A))
l2ALoss = tf.reduce_mean(tf.square(A))
e1Term = tf.multiply(elasticParam1, l1ALoss)
e2Term = tf.multiply(elasticParam2, l2ALoss)
loss = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(Ytarget - res)), e1Term), e2Term), 0)

# Initialize variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Declare optimizer
opt = tf.train.GradientDescentOptimizer(0.001)
step = opt.minimize(loss)

# Training loop
lossVec = []
for i in range(1000):
    index = np.random.choice(len(X), size=bSize)
    x = X[index]
    y = np.transpose([Y[index]])
    sess.run(step, feed_dict={Xdata: x, Ytarget: y})
    ls = sess.run(loss, feed_dict={Xdata: x, Ytarget: y})
    lossVec.append(ls[0])
    if (i+1)%250==0:
        print('Step : ' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Loss = ' + str(ls))

# Get the optimal coefficients
[[sw_coef], [pl_coef], [pw_ceof]] = sess.run(A)
[intercept] = sess.run(b)

# Plot loss over time
plt.plot(lossVec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
