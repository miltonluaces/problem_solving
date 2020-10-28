# Using a Multiple Layer Network
#---------------------------------------
#
# We will illustrate how to use a Multiple
# Layer Network in Tensorflow
#
# Low Birthrate data:
#
#Columns    Variable                                              Abbreviation
#-----------------------------------------------------------------------------
# Identification Code                                     ID
# Low Birth Weight (0 = Birth Weight >= 2500g,            LOW
#                          1 = Birth Weight < 2500g)
# Age of the Mother in Years                              AGE
# Weight in Pounds at the Last Menstrual Period           LWT
# Race (1 = White, 2 = Black, 3 = Other)                  RACE
# Smoking Status During Pregnancy (1 = Yes, 0 = No)       SMOKE
# History of Premature Labor (0 = None  1 = One, etc.)    PTL
# History of Hypertension (1 = Yes, 0 = No)               HT
# Presence of Uterine Irritability (1 = Yes, 0 = No)      UI
# Number of Physician Visits During the First Trimester   FTV
#                (0 = None, 1 = One, 2 = Two, etc.)
# Birth Weight in Grams                                   BWT
#------------------------------
# The multiple neural network layer we will create will be composed of
# three fully connected hidden layers, with node sizes 25, 10, and 3

import tensorflow as tf
import matplotlib.pyplot as plt
import requests
import numpy as np


# Set Seed
seed = 3
tf.set_random_seed(seed)
np.random.seed(seed)


birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\r\n')[5:]
birth_header = [x for x in birth_data[0].split(' ') if len(x)>=1]
birth_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in birth_data[1:] if len(y)>=1]


bSize = 100

# Extract y-target (birth weight)
Y = np.array([x[10] for x in birth_data])

# Filter for features of interest
cols_of_interest = ['AGE', 'LWT', 'RACE', 'SMOKE', 'PTL', 'HT', 'UI', 'FTV']
X = np.array([[x[ix] for ix, feature in enumerate(birth_header) if feature in cols_of_interest] for x in birth_data])

# Create graph session 
sess = tf.Session()

# Split data into train/test = 80%/20%
trainIdx = np.random.choice(len(X), round(len(X)*0.8), replace=False)
testIdx = np.array(list(set(range(len(X))) - set(trainIdx)))
Xtrain = X[trainIdx]
Xtest = X[testIdx]
Ytrain = Y[trainIdx]
Ytest = Y[testIdx]


# Normalize by column (min-max norm to be between 0 and 1)
def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m-col_min) / (col_max - col_min)
    
Xtrain = np.nan_to_num(normalize_cols(Xtrain))
Xtest = np.nan_to_num(normalize_cols(Xtest))


# Define Variable Functions (weights and bias)
def init_weight(shape, st_dev):
    weight = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return(weight)
    

def init_bias(shape, st_dev):
    bias = tf.Variable(tf.random_normal(shape, stddev=st_dev))
    return(bias)
    
    
# Create Placeholders
Xdata = tf.placeholder(shape=[None, 8], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[None, 1], dtype=tf.float32)


# Create a fully connected layer:
def fully_connected(input_layer, weights, biases):
    layer = tf.add(tf.matmul(input_layer, weights), biases)
    return(tf.nn.relu(layer))


#--------Create the first layer (25 hidden nodes)--------
weight_1 = init_weight(shape=[8, 25], st_dev=10.0)
bias_1 = init_bias(shape=[25], st_dev=10.0)
layer_1 = fully_connected(Xdata, weight_1, bias_1)

#--------Create second layer (10 hidden nodes)--------
weight_2 = init_weight(shape=[25, 10], st_dev=10.0)
bias_2 = init_bias(shape=[10], st_dev=10.0)
layer_2 = fully_connected(layer_1, weight_2, bias_2)


#--------Create third layer (3 hidden nodes)--------
weight_3 = init_weight(shape=[10, 3], st_dev=10.0)
bias_3 = init_bias(shape=[3], st_dev=10.0)
layer_3 = fully_connected(layer_2, weight_3, bias_3)


#--------Create output layer (1 output value)--------
weight_4 = init_weight(shape=[3, 1], st_dev=10.0)
bias_4 = init_bias(shape=[1], st_dev=10.0)
final_output = fully_connected(layer_3, weight_4, bias_4)

# Declare loss function (L1)
loss = tf.reduce_mean(tf.abs(Ytarget - final_output))

# Declare optimizer
opt = tf.train.AdamOptimizer(0.05)
step = opt.minimize(loss)

# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
lossVec = []
testLoss = []
for i in range(200):
    index = np.random.choice(len(Xtrain), size=bSize)
    x = Xtrain[index]
    y = np.transpose([Ytrain[index]])
    sess.run(step, feed_dict={Xdata: x, Ytarget: y})

    ls = sess.run(loss, feed_dict={Xdata: x, Ytarget: y})
    lossVec.append(ls)
    
    test_ls = sess.run(loss, feed_dict={Xdata: Xtest, Ytarget: np.transpose([Ytest])})
    testLoss.append(test_ls)
    if (i+1)%25==0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str(ls))


# Plot loss over time
plt.plot(lossVec, 'k-', label='Train Loss')
plt.plot(testLoss, 'r--', label='Test Loss')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc="upper right")
plt.show()

# Find the % classified correctly above/below the cutoff of 2500 g
# >= 2500 g = 0
# < 2500 g = 1
actuals = np.array([x[1] for x in birth_data])
test_actuals = actuals[testIdx]
train_actuals = actuals[trainIdx]

test_preds = [x[0] for x in sess.run(final_output, feed_dict={Xdata: Xtest})]
train_preds = [x[0] for x in sess.run(final_output, feed_dict={Xdata: Xtrain})]
test_preds = np.array([1.0 if x<2500.0 else 0.0 for x in test_preds])
train_preds = np.array([1.0 if x<2500.0 else 0.0 for x in train_preds])

# Print out accuracies
test_acc = np.mean([x==y for x,y in zip(test_preds, test_actuals)])
train_acc = np.mean([x==y for x,y in zip(train_preds, train_actuals)])
print('On predicting the category of low birthweight from regression output (<2500g):')
print('Test Accuracy: {}'.format(test_acc))
print('Train Accuracy: {}'.format(train_acc))