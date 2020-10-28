# Combining Everything Together
#------------------------------

# Binary classification on the class if iris dataset. We will only predict if a flower is I.setosa or not.

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf


# Load the iris data
# iris.target = {0, 1, 2}, where '0' is setosa
# iris.data ~ [sepal.width, sepal.length, pedal.width, pedal.length]
iris = datasets.load_iris()
binarYtarget = np.array([1. if x==0 else 0. for x in iris.target])
iris2d = np.array([[x[2], x[3]] for x in iris.data])

# Declare batch size
bSize = 20

# Declare placeholders
x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables A and b (0 = x1 - A*x2 + b)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Add model to graph: x1 - A*x2 + b
prod = tf.matmul(x2_data, A)
add = tf.add(prod, b)
res = tf.sub(x1_data, add)
#res = tf.sub(Xdata[0], tf.add(tf.matmul(Xdata[1], A), b))

# Add classification loss (cross entropy)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(res, Ytarget)

# Create Optimizer
opt = tf.train.GradientDescentOptimizer(0.05)
step = opt.minimize(xentropy)

# Initialize
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Run
for i in range(1000):
    index = np.random.choice(len(iris2d), size=bSize)
    #x = np.transpose([iris_2d[index]])
    x = iris2d[index]
    x1 = np.array([[x[0]] for x in x])
    x2 = np.array([[x[1]] for x in x])
    #y = np.transpose([binarYtarget[index]])
    y = np.array([[y] for y in binarYtarget[index]])
    sess.run(step, feed_dict={x1_data: x1, x2_data: x2, Ytarget: y})
    if (i+1)%200 == 0:
        print('Step : ' + str(i+1) + ' A = ' + str(sess.run(A)) + ', b = ' + str(sess.run(b)))
        

# Print Results : Pull out slope/intercept
[[slope]] = sess.run(A)
[[intercept]] = sess.run(b)

# Create fitted line
x = np.linspace(0, 3, num=50)
ablineValues = []
for i in x:
  ablineValues.append(slope*i+intercept)

# Plot the fitted line over the data
setosa_x = [a[1] for i,a in enumerate(iris2d) if binarYtarget[i]==1]
setosa_y = [a[0] for i,a in enumerate(iris2d) if binarYtarget[i]==1]
non_setosa_x = [a[1] for i,a in enumerate(iris2d) if binarYtarget[i]==0]
non_setosa_y = [a[0] for i,a in enumerate(iris2d) if binarYtarget[i]==0]
plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='setosa')
plt.plot(non_setosa_x, non_setosa_y, 'ro', label='Non-setosa')
plt.plot(x, ablineValues, 'b-')
plt.xlim([0.0, 2.7])
plt.ylim([0.0, 7.1])
plt.suptitle('Linear Separator For I.setosa', fontsize=20)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='lower right')
plt.show()