# Improving Linear Regression with Neural Networks (Logistic Regression)
#----------------------------------
#
# This function shows how to use Tensorflow to
# solve logistic regression with a multiple layer neural network
# y = sigmoid(A3 * sigmoid(A2* sigmoid(A1*x + b1) + b2) + b3)
#
# We will use the low birth weight data, specifically:
#  y = 0 or 1 = low birth weight
#  x = demographic and medical history data

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests


# Create graph
sess = tf.Session()

birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\r\n')[5:]
birth_header = [x for x in birth_data[0].split(' ') if len(x)>=1]
birth_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in birth_data[1:] if len(y)>=1]
# Pull out target variable
Y = np.array([x[1] for x in birth_data])
# Pull out predictor variables (not id, not target, and not birthweight)
X = np.array([x[2:9] for x in birth_data])

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
bSize = 90

# Initialize placeholders
Xdata = tf.placeholder(shape=[None, 7], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[None, 1], dtype=tf.float32)


# Create variable definition
def init_variable(shape):
    return(tf.Variable(tf.random_normal(shape=shape)))


# Create a logistic layer definition
def logistic(input_layer, multiplication_weight, bias_weight, activation = True):
    linear_layer = tf.add(tf.matmul(input_layer, multiplication_weight), bias_weight)
    # We separate the activation at the end because the loss function will
    # implement the last sigmoid necessary
    if activation:
        return(tf.nn.sigmoid(linear_layer))
    else:
        return(linear_layer)


# First logistic layer (7 inputs to 7 hidden nodes)
A1 = init_variable(shape=[7,14])
b1 = init_variable(shape=[14])
logistic_layer1 = logistic(Xdata, A1, b1)

# Second logistic layer (7 hidden inputs to 5 hidden nodes)
A2 = init_variable(shape=[14,5])
b2 = init_variable(shape=[5])
logistic_layer2 = logistic(logistic_layer1, A2, b2)

# Final output layer (5 hidden nodes to 1 output)
A3 = init_variable(shape=[5,1])
b3 = init_variable(shape=[1])
final_output = logistic(logistic_layer2, A3, b3, activation=False)

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(final_output, Ytarget))

# Declare optimizer
opt = tf.train.AdamOptimizer(learning_rate = 0.005)
step = opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Actual Prediction
prediction = tf.round(tf.nn.sigmoid(final_output))
predictions_correct = tf.cast(tf.equal(prediction, Ytarget), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

# Training loop
lossVec = []
train_acc = []
test_acc = []
for i in range(1500):
    index = np.random.choice(len(Xtrain), size=bSize)
    x = Xtrain[index]
    y = np.transpose([Ytrain[index]])
    sess.run(step, feed_dict={Xdata: x, Ytarget: y})

    ls = sess.run(loss, feed_dict={Xdata: x, Ytarget: y})
    lossVec.append(ls)
    temp_acc_train = sess.run(accuracy, feed_dict={Xdata: Xtrain, Ytarget: np.transpose([Ytrain])})
    train_acc.append(temp_acc_train)
    temp_acc_test = sess.run(accuracy, feed_dict={Xdata: Xtest, Ytarget: np.transpose([Ytest])})
    test_acc.append(temp_acc_test)
    if (i+1)%150==0:
        print('Loss = ' + str(ls))
        
# Plot loss over time
plt.plot(lossVec, 'k-')
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.show()

# Plot train and test accuracy
plt.plot(train_acc, 'k-', label='Train Set Accuracy')
plt.plot(test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()