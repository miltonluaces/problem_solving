# SVM Regression
#----------------------------------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

# Load the data (iris.data) = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
X = np.array([x[3] for x in iris.data])
Y = np.array([y[0] for y in iris.data])

# Split data train/test
trainIdx = np.random.choice(len(X), round(len(X)*0.8), replace=False)
testIdx = np.array(list(set(range(len(X))) - set(trainIdx)))
Xtrain = X[trainIdx]
Xtest = X[testIdx]
Ytrain = Y[trainIdx]
Ytest = Y[testIdx]

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

# Declare loss function = max(0, abs(target - predicted) + epsilon).  1/2 margin width parameter = epsilon. The aim is to find the line that has the maximum margin which INCLUDES as many points as possible
epsilon = tf.constant([0.5])
loss = tf.reduce_mean(tf.maximum(0., tf.sub(tf.abs(tf.sub(res, Ytarget)), epsilon)))

# Declare optimizer
opt = tf.train.GradientDescentOptimizer(0.075)
step = opt.minimize(loss)

# Initialize variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
trainLoss = []
testLoss = []
for i in range(200):
    index = np.random.choice(len(Xtrain), size=bSize)
    x = np.transpose([Xtrain[index]])
    y = np.transpose([Ytrain[index]])
    sess.run(step, feed_dict={Xdata: x, Ytarget: y})
    
    temp_trainLoss = sess.run(loss, feed_dict={Xdata: np.transpose([Xtrain]), Ytarget: np.transpose([Ytrain])})
    trainLoss.append(temp_trainLoss)
    
    temp_testLoss = sess.run(loss, feed_dict={Xdata: np.transpose([Xtest]), Ytarget: np.transpose([Ytest])})
    testLoss.append(temp_testLoss)
    if (i+1)%50 == 0:
        print('-----------')
        print('Generation: ' + str(i+1))
        print('A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
        print('Train Loss = ' + str(temp_trainLoss))
        print('Test Loss = ' + str(temp_testLoss))

# Extract Coefficients
[[slope]] = sess.run(A)
[[y_intercept]] = sess.run(b)
[width] = sess.run(epsilon)

# Get best fit line
best = []
bestUpr = []
bestLwr = []
for i in X:
  best.append(slope*i+y_intercept)
  bestUpr.append(slope*i+y_intercept+width)
  bestLwr.append(slope*i+y_intercept-width)

# Plot fit with data
plt.plot(X, Y, 'o', label='Data Points')
plt.plot(X, best, 'r-', label='SVM Regression Line', linewidth=3)
plt.plot(X, bestUpr, 'r--', linewidth=2)
plt.plot(X, bestLwr, 'r--', linewidth=2)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()

# Plot loss over time
plt.plot(trainLoss, 'k-', label='Train Set Loss')
plt.plot(testLoss, 'r--', label='Test Set Loss')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.legend(loc='upper right')
plt.show()
