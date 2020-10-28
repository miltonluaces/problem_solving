# Logistic Regression
#----------------------------------
#
# This function shows how to use Tensorflow to
# solve logistic regression.
# y = sigmoid(Ax + b)
#
# We will use the low birth weight data, specifically:
#  y = 0 or 1 = low birth weight
#  x = demographic and medical history data

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests
from sklearn import datasets
from sklearn.preprocessing import normalize
import csv as cs


# Load data
#dataUrl = 'http://faculty.washington.edu/heagerty/Courses/b513/WEB2002/datasets/lowbwt.dat'
#dataFile = requests.get(dataUrl)
#data = dataFile.text.split()[5:]
#header = [x for x in data[0].split(' ') if len(x)>=1]
#data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in data[1:] if len(y)>=1]
#data = np.loadtxt(open(dataFile, "rb"), delimiter=" ", skiprows=0)
#text = dataFile.text
reader = cs.reader(open("lowbwt.txt", "rb"), delimiter=' ')
x = np.loadtxt(open("lowbwt.txt", "rb"), delimiter=" ")
#x = list(reader)
result = np.array(x).astype("float")

# Pull out target variable
Y = np.array([x[1] for x in data])
# Pull out predictor variables (not id, not target, and not birthweight)
X = np.array([x[2:9] for x in data])

# Split data into train/test = 80%/20%
trainIdx = np.random.choice(len(X), round(len(X)*0.8), replace=False)
testIdx = np.array(list(set(range(len(X))) - set(trainIdx)))
Xtrain = X[trainIdx]
Xtest = X[testIdx]
Ytrain = Y[trainIdx]
Ytest = Y[testIdx]

# Normalize by column (min-max norm)
def NormalizeCols(m):
    colMax = m.max(axis=0)
    colMin = m.min(axis=0)
    return (m-colMin) / (colMax-colMin)
    
Xtrain = np.nan_to_num(NormalizeCols(Xtrain))
Xtest = np.nan_to_num(NormalizeCols(Xtest))

# Declare batch size
bSize = 25

# Initialize placeholders
Xdata = tf.placeholder(shape=[None, 7], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[7,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Declare model operations
res = tf.add(tf.matmul(Xdata, A), b)

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=res, logits=Ytarget))

# Initialize variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Declare optimizer
opt = tf.train.GradientDescentOptimizer(0.01)
step = opt.minimize(loss)

# Actual Prediction
pred = tf.round(tf.sigmoid(res))
hits = tf.cast(tf.equal(pred, Ytarget), tf.float32)
accuracy = tf.reduce_mean(hits)

# Training loop
lossVec = []
trainAcc = []
testAcc = []
for i in range(1500):
    index = np.random.choice(len(Xtrain), size=bSize)
    x = Xtrain[index]
    y = np.transpose([Ytrain[index]])
    sess.run(step, feed_dict={Xdata: x, Ytarget: y})

    ls = sess.run(loss, feed_dict={Xdata: x, Ytarget: y})
    lossVec.append(ls)
    temp_acc_train = sess.run(accuracy, feed_dict={Xdata: Xtrain, Ytarget: np.transpose([Ytrain])})
    trainAcc.append(temp_acc_train)
    temp_acc_test = sess.run(accuracy, feed_dict={Xdata: Xtest, Ytarget: np.transpose([Ytest])})
    testAcc.append(temp_acc_test)
    if (i+1)%300==0:
        print('Loss = ' + str(ls))
        
# Plot loss over time
plt.plot(lossVec, 'k-')
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.show()

# Plot train and test accuracy
plt.plot(trainAcc, 'k-', label='Train Set Accuracy')
plt.plot(testAcc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()