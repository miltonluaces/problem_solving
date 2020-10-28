# MNIST Digit Prediction with k-Nearest Neighbors
#-----------------------------------------------
#
# This script will load the MNIST data, and split
# it into test/train and perform prediction with
# nearest neighbors
#
# For each test integer, we will return the
# closest image/integer.
#
# Integer images are represented as 28x8 matrices
# of floating point numbers

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data


# Create graph
sess = tf.Session()

# Load the data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Random sample
train_size = 1000
test_size = 102
rand_trainIdx = np.random.choice(len(mnist.train.images), train_size, replace=False)
rand_testIdx = np.random.choice(len(mnist.test.images), test_size, replace=False)
Xtrain = mnist.train.images[rand_trainIdx]
Xtest = mnist.test.images[rand_testIdx]
Ytrain = mnist.train.labels[rand_trainIdx]
Ytest = mnist.test.labels[rand_testIdx]

# Declare k-value and batch size
k = 4
bSize=6

# Placeholders
Xdata_train = tf.placeholder(shape=[None, 784], dtype=tf.float32)
Xdata_test = tf.placeholder(shape=[None, 784], dtype=tf.float32)
Ytarget_train = tf.placeholder(shape=[None, 10], dtype=tf.float32)
Ytarget_test = tf.placeholder(shape=[None, 10], dtype=tf.float32)

# Declare distance metric
# L1
distance = tf.reduce_sum(tf.abs(tf.sub(Xdata_train, tf.expand_dims(Xdata_test,1))), reduction_indices=2)

# L2
#distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(Xdata_train, tf.expand_dims(Xdata_test,1))), reduction_indices=1))

# Predict: Get min distance index (Nearest neighbor)
top_k_xvals, top_k_indices = tf.nn.top_k(tf.neg(distance), k=k)
prediction_indices = tf.gather(Ytarget_train, top_k_indices)
# Predict the mode category
count_of_predictions = tf.reduce_sum(prediction_indices, reduction_indices=1)
prediction = tf.argmax(count_of_predictions, dimension=1)

# Calculate how many loops over training data
num_loops = int(np.ceil(len(Xtest)/bSize))

test_output = []
actual_vals = []
for i in range(num_loops):
    min_index = i*bSize
    max_index = min((i+1)*bSize,len(Xtrain))
    x_batch = Xtest[min_index:max_index]
    y_batch = Ytest[min_index:max_index]
    predictions = sess.run(prediction, feed_dict={Xdata_train: Xtrain, Xdata_test: x_batch,
                                         Ytarget_train: Ytrain, Ytarget_test: y_batch})
    test_output.extend(predictions)
    actual_vals.extend(np.argmax(y_batch, axis=1))

accuracy = sum([1./test_size for i in range(test_size) if test_output[i]==actual_vals[i]])
print('Accuracy on test set: ' + str(accuracy))

# Plot the last batch results:
actuals = np.argmax(y_batch, axis=1)

Nrows = 2
Ncols = 3
for i in range(len(actuals)):
    plt.subplot(Nrows, Ncols, i+1)
    plt.imshow(np.reshape(x_batch[i], [28,28]), cmap='Greys_r')
    plt.title('Actual: ' + str(actuals[i]) + ' Pred: ' + str(predictions[i]),
                               fontsize=10)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)