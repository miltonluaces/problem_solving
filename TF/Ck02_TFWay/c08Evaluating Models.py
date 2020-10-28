# Evaluating models
# -----------------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Regression Example: X: 100 random samples from a normal ~ N(1, 0.1). Y: 100 values of the value 10. We will fit the model: Y = A

# Declare batch size
bSize = 25

# Create data
X = np.random.normal(1, 0.1, 100)
Y = np.repeat(10., 100)
Xdata = tf.placeholder(shape=[None, 1], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Split data into train/test = 80%/20%
trainIdx = np.random.choice(len(X), round(len(X)*0.8), replace=False)
testIdx = np.array(list(set(range(len(X))) - set(trainIdx)))
Xtrain = X[trainIdx]
Xtest = X[testIdx]
Ytrain = Y[trainIdx]
Ytest = Y[testIdx]

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
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Run 
for i in range(100):
    index = np.random.choice(len(Xtrain), size=bSize)
    x = np.transpose([Xtrain[index]])
    y = np.transpose([Ytrain[index]])
    sess.run(step, feed_dict={Xdata: x, Ytarget: y})
    if (i+1)%25 == 0:
        print('Step : ' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={Xdata: x, Ytarget: y})))

# Evaluate accuracy (loss) on test set
mse_test = sess.run(loss, feed_dict={Xdata: np.transpose([Xtest]), Ytarget: np.transpose([Ytest])})
mse_train = sess.run(loss, feed_dict={Xdata: np.transpose([Xtrain]), Ytarget: np.transpose([Ytrain])})
print('MSE on test:' + str(np.round(mse_test, 2)))
print('MSE on train:' + str(np.round(mse_train, 2)))

# Classification Example
# X: 50 samples from a normal = N(-1, 1) + 50 samples from a normal = N(1, 1)
# target: 50 values of 0 + 50 values of 1.
# We will fit the binary classification model: if sigmoid(x+A) < 0.5 -> 0 else 1 ( A should be -(mean1 + mean2)/2 )

# Declare batch size
bSize = 25

# Create data
X = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(2, 1, 50)))
Y = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
Xdata = tf.placeholder(shape=[1, None], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[1, None], dtype=tf.float32)

# Split data into train/test = 80%/20%
trainIdx = np.random.choice(len(X), round(len(X)*0.8), replace=False)
testIdx = np.array(list(set(range(len(X))) - set(trainIdx)))
Xtrain = X[trainIdx]
Xtest = X[testIdx]
Ytrain = Y[trainIdx]
Ytest = Y[testIdx]

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

# Add operation to graph : Want to create the operation sigmoid(x + A). Note, the sigmoid() part is in the loss function
res = tf.add(Xdata, A)

# Add classification loss (cross entropy)
xentropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(res, Ytarget))

# Create Optimizer
opt = tf.train.GradientDescentOptimizer(0.05)
step = opt.minimize(xentropy)

# Initialize 
tf.reset_default_graph()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Run 
for i in range(1800):
    index = np.random.choice(len(Xtrain), size=bSize)
    x = [Xtrain[index]]
    y = [Ytrain[index]]
    sess.run(step, feed_dict={Xdata: x, Ytarget: y})
    if (i+1)%200 == 0:
        print('Step : ' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy, feed_dict={Xdata: x, Ytarget: y})))
        
# Evaluate Predictions on test set
yPred = tf.squeeze(tf.round(tf.nn.sigmoid(tf.add(Xdata, A))))
hits = tf.equal(yPred, Ytarget)
accuracy = tf.reduce_mean(tf.cast(hits, tf.float32))
accTest = sess.run(accuracy, feed_dict={Xdata: [Xtest], Ytarget: [Ytest]})
accTrain = sess.run(accuracy, feed_dict={Xdata: [Xtrain], Ytarget: [Ytrain]})
print('Accuracy on train set: ' + str(acc_value_train))
print('Accuracy on test set: ' + str(accTest))

# Plot classification result
Aval = -sess.run(A)
bins = np.linspace(-5, 5, 50)
plt.hist(X[0:50], bins, alpha=0.5, label='N(-1,1)', color='white')
plt.hist(X[50:100], bins[0:50], alpha=0.5, label='N(2,1)', color='red')
plt.plot((Aval, Aval), (0, 8), 'k--', linewidth=3, label='A = '+ str(np.round(Aval, 2)))
plt.legend(loc='upper right')
plt.title('Binary Classifier, Accuracy=' + str(np.round(acc_value, 2)))
plt.show()