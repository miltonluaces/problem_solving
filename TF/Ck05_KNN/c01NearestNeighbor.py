# k-Nearest Neighbor
#----------------------------------
#
# This function illustrates how to use
# k-nearest neighbors in tensorflow
#
# We will use the 1970s Boston housing dataset
# which is available through the UCI
# ML data repository.
#
# Data:
#----------x-values-----------
# CRIM   : per capita crime rate by town
# ZN     : prop. of res. land zones
# INDUS  : prop. of non-retail business acres
# CHAS   : Charles river dummy variable
# NOX    : nitrix oxides concentration / 10 M
# RM     : Avg. # of rooms per building
# AGE    : prop. of buildings built prior to 1940
# DIS    : Weighted distances to employment centers
# RAD    : Index of radian highway access
# TAX    : Full tax rate value per $10k
# PTRATIO: Pupil/Teacher ratio by town
# B      : 1000*(Bk-0.63)^2, Bk=prop. of blacks
# LSTAT  : % lower status of pop
#------------y-value-----------
# MEDV   : Median Value of homes in $1,000's

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests


# Create graph
sess = tf.Session()

# Load the data
housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
cols_used = ['CRIM', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT']
num_features = len(cols_used)
housing_file = requests.get(housing_url)
housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]

Y = np.transpose([np.array([y[13] for y in housing_data])])
X = np.array([[x for i,x in enumerate(y) if housing_header[i] in cols_used] for y in housing_data])

## Min-Max Scaling
X = (X - X.min(0)) / X.ptp(0)

# Split the data into train and test sets
trainIdx = np.random.choice(len(X), round(len(X)*0.8), replace=False)
testIdx = np.array(list(set(range(len(X))) - set(trainIdx)))
Xtrain = X[trainIdx]
Xtest = X[testIdx]
Ytrain = Y[trainIdx]
Ytest = Y[testIdx]

# Declare k-value and batch size
k = 4
bSize=len(Xtest)

# Placeholders
Xdata_train = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
Xdata_test = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
Ytarget_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
Ytarget_test = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Declare distance metric
# L1
distance = tf.reduce_sum(tf.abs(tf.sub(Xdata_train, tf.expand_dims(Xdata_test,1))), reduction_indices=2)

# L2
#distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(Xdata_train, tf.expand_dims(Xdata_test,1))), reduction_indices=1))

# Predict: Get min distance index (Nearest neighbor)
top_k_xvals, top_k_indices = tf.nn.top_k(tf.neg(distance), k=k)
x_sums = tf.expand_dims(tf.reduce_sum(top_k_xvals, 1),1)
x_sums_repeated = tf.matmul(x_sums,tf.ones([1, k], tf.float32))
x_val_weights = tf.expand_dims(tf.div(top_k_xvals,x_sums_repeated), 1)

top_k_yvals = tf.gather(Ytarget_train, top_k_indices)
prediction = tf.squeeze(tf.batch_matmul(x_val_weights,top_k_yvals), squeeze_dims=[1])

# Calculate MSE
mse = tf.div(tf.reduce_sum(tf.square(tf.sub(prediction, Ytarget_test))), bSize)

# Calculate how many loops over training data
num_loops = int(np.ceil(len(Xtest)/bSize))

for i in range(num_loops):
    min_index = i*bSize
    max_index = min((i+1)*bSize,len(Xtrain))
    x_batch = Xtest[min_index:max_index]
    y_batch = Ytest[min_index:max_index]
    predictions = sess.run(prediction, feed_dict={Xdata_train: Xtrain, Xdata_test: x_batch,
                                         Ytarget_train: Ytrain, Ytarget_test: y_batch})
    batch_mse = sess.run(mse, feed_dict={Xdata_train: Xtrain, Xdata_test: x_batch,
                                         Ytarget_train: Ytrain, Ytarget_test: y_batch})

    print('Batch #' + str(i+1) + ' MSE: ' + str(np.round(batch_mse,3)))

# Plot prediction and actual distribution
bins = np.linspace(5, 50, 45)

plt.hist(predictions, bins, alpha=0.5, label='Prediction')
plt.hist(y_batch, bins, alpha=0.5, label='Actual')
plt.title('Histogram of Predicted and Actual Values')
plt.xlabel('Med Home Value in $1,000s')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()

