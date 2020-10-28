# Illustration of Various Kernels
#--------------------------------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

# Generate non-lnear data
(X, Y) = datasets.make_circles(n_samples=350, factor=.5, noise=.1)
Y = np.array([1 if y==1 else -1 for y in Y])
class1X = [x[0] for i,x in enumerate(X) if Y[i]==1]
class1Y = [x[1] for i,x in enumerate(X) if Y[i]==1]
class2X = [x[0] for i,x in enumerate(X) if Y[i]==-1]
class2Y = [x[1] for i,x in enumerate(X) if Y[i]==-1]

# Declare batch size
bSize = 350

# Initialize placeholders
Xdata = tf.placeholder(shape=[None, 2], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[None, 1], dtype=tf.float32)
predGrid = tf.placeholder(shape=[None, 2], dtype=tf.float32)

# Create variables for svm
b = tf.Variable(tf.random_normal(shape=[1,bSize]))

# Linear Kernel: K(x1, x2) = t(x1) * x2
linKernel = tf.matmul(Xdata, tf.transpose(Xdata))

# Gaussian (RBF) kernel: K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)
gamma = tf.constant(-50.0)
dist = tf.reduce_sum(tf.square(Xdata), 1)
dist = tf.reshape(dist, [-1,1])
sq_dists = tf.add(tf.sub(dist, tf.multiply(2., tf.matmul(Xdata, tf.transpose(Xdata)))), tf.transpose(dist))
gauKernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

# Compute SVM Model
res = tf.matmul(b, gauKernel)
first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
Ytarget_cross = tf.matmul(Ytarget, tf.transpose(Ytarget))
second_term = tf.reduce_sum(tf.multiply(gauKernel, tf.multiply(b_vec_cross, Ytarget_cross)))
loss = tf.neg(tf.sub(first_term, second_term))

# Create Prediction Kernel
# Linear prediction kernel
linKernel = tf.matmul(Xdata, tf.transpose(predGrid))

# Gaussian (RBF) prediction kernel
rA = tf.reshape(tf.reduce_sum(tf.square(Xdata), 1),[-1,1])
rB = tf.reshape(tf.reduce_sum(tf.square(predGrid), 1),[-1,1])
predSqDist = tf.add(tf.sub(rA, tf.multiply(2., tf.matmul(Xdata, tf.transpose(predGrid)))), tf.transpose(rB))
predKernel = tf.exp(tf.multiply(gamma, tf.abs(predSqDist)))

predOutput = tf.matmul(tf.multiply(tf.transpose(Ytarget),b), predKernel)
pred = tf.sign(predOutput-tf.reduce_mean(predOutput))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(pred), tf.squeeze(Ytarget)), tf.float32))

# Declare optimizer
opt = tf.train.GradientDescentOptimizer(0.002)
step = opt.minimize(loss)

# Initialize variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
lossVec = []
batchAcc = []
for i in range(1000):
    index = np.random.choice(len(X), size=bSize)
    x = X[index]
    y = np.transpose([Y[index]])
    sess.run(step, feed_dict={Xdata: x, Ytarget: y})
    
    ls = sess.run(loss, feed_dict={Xdata: x, Ytarget: y})
    lossVec.append(ls)
    
    acc_temp = sess.run(accuracy, feed_dict={Xdata: x,
                                             Ytarget: y,
                                             predGrid:x})
    batchAcc.append(acc_temp)
    
    if (i+1)%250 == 0:
        print('Step : ' + str(i+1))
        print('Loss = ' + str(ls))

# Create a mesh to plot points in
xMin, xMax = X[:, 0].min() - 1, X[:, 0].max() + 1
yMin, yMax = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.02),
                     np.arange(yMin, yMax, 0.02))
gridPoints = np.c_[xx.ravel(), yy.ravel()]
[gridPreds] = sess.run(pred, feed_dict={Xdata: x,
                                                   Ytarget: y,
                                                   predGrid: gridPoints})
gridPreds = gridPred.reshape(xx.shape)

# Plot points and grid
plt.contourf(xx, yy, gridPreds, cmap=plt.cm.Paired, alpha=0.8)
plt.plot(class1X, class1Y, 'ro', label='Class 1')
plt.plot(class2X, class2Y, 'kx', label='Class -1')
plt.title('Gaussian SVM Results')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='lower right')
plt.ylim([-1.5, 1.5])
plt.xlim([-1.5, 1.5])
plt.show()

# Plot batch accuracy
plt.plot(batchAcc, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss over time
plt.plot(lossVec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

# sess.run(predOutput, feed_dict={Xdata: x, Ytarget: y, prediction_grid: grid_points})
# sess.run(predKernel, feed_dict={Xdata: x, Ytarget: y, prediction_grid: grid_points})
# sess.run(res, feed_dict={Xdata:x, Ytarget: y})
# sess.run(second_term, feed_dict={Xdata:x, Ytarget: y})
