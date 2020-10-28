# Implementing a one-layer Neural Network
#---------------------------------------
#
# We will illustrate how to create a one hidden layer NN
#
# We will use the iris data for this exercise
#
# We will build a one-hidden layer neural network
#  to predict the fourth attribute, Petal Width from
#  the other three (Sepal length, Sepal width, Petal length).

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets


iris = datasets.load_iris()
X = np.array([x[0:3] for x in iris.data])
Y = np.array([x[3] for x in iris.data])

# Create graph session 
sess = tf.Session()

# Set Seed
seed = 3
tf.set_random_seed(seed)
np.random.seed(seed)

# Split data into train/test = 80%/20%
trainIdx = np.random.choice(len(X), round(len(X)*0.8), replace=False)
testIdx = np.array(list(set(range(len(X))) - set(trainIdx)))
Xtrain = X[trainIdx]
Xtest = X[testIdx]
Ytrain = Y[trainIdx]
Ytest = Y[testIdx]

# Normalize by column (min-max norm)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)
    
Xtrain = np.nan_to_num(normalize_cols(Xtrain))
Xtest = np.nan_to_num(normalize_cols(Xtest))

# Declare batch size
bSize = 50

# Initialize placeholders
Xdata = tf.placeholder(shape=[None, 3], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for both Neural Network Layers
hidden_layer_nodes = 10
A1 = tf.Variable(tf.random_normal(shape=[3,hidden_layer_nodes])) # inputs -> hidden nodes
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))   # one biases for each hidden node
A2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,1])) # hidden inputs -> 1 output
b2 = tf.Variable(tf.random_normal(shape=[1]))   # 1 bias for the output


# Declare model operations
hidden_output = tf.nn.relu(tf.add(tf.matmul(Xdata, A1), b1))
final_output = tf.nn.relu(tf.add(tf.matmul(hidden_output, A2), b2))

# Declare loss function
loss = tf.reduce_mean(tf.square(Ytarget - final_output))

# Declare optimizer
opt = tf.train.GradientDescentOptimizer(0.005)
step = opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
lossVec = []
testLoss = []
for i in range(500):
    index = np.random.choice(len(Xtrain), size=bSize)
    x = Xtrain[index]
    y = np.transpose([Ytrain[index]])
    sess.run(step, feed_dict={Xdata: x, Ytarget: y})

    ls = sess.run(loss, feed_dict={Xdata: x, Ytarget: y})
    lossVec.append(np.sqrt(ls))
    
    test_ls = sess.run(loss, feed_dict={Xdata: Xtest, Ytarget: np.transpose([Ytest])})
    testLoss.append(np.sqrt(test_ls))
    if (i+1)%50==0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str(ls))


# Plot loss (MSE) over time
plt.plot(lossVec, 'k-', label='Train Loss')
plt.plot(testLoss, 'r--', label='Test Loss')
plt.title('Loss (MSE) per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()