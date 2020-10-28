# Back Propagation
#----------------------------------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Regression Example:
# X: 100 samples from a normal ~ N(1, 0.1)
# Y: 100 values of the value 10.
# We will fit the model: Y = A * X (theoretically, A = 10)

# Create data
X = np.random.normal(1, 0.1, 100)
Y = np.repeat(10., 100)
Xdata = tf.placeholder(shape=[1], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[1], dtype=tf.float32)

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(shape=[1]))

# Add operation to graph
res = tf.multiply(Xdata, A)

# Add L2 loss operation to graph
loss = tf.square(res - Ytarget)

# Create Optimizer
opt = tf.train.GradientDescentOptimizer(0.02)
step = opt.minimize(loss)

# Initialize 
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Run 
for i in range(100):
    index = np.random.choice(100)
    x = [X[index]]
    y = [Y[index]]
    tf.reset_default_graph()
    sess.run(step, feed_dict={Xdata: x, Ytarget: y})
    if (i+1)%25 == 0:
        print('Step : ' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(loss, feed_dict={Xdata: x, Ytarget: y})))

# Classification Example
# We will create sample data as follows:
# X: 100 samples from a normal = N(-1, 1) + 50 samples from a normal = N(1, 1)
# Y: 50 values of 0 + 50 values of 1.
# We will fit the binary classification model: if sigmoid(x+A) < 0.5 -> 0 else 1  (A should be -(mean1 + mean2)/2 )

# Create data
X = np.concatenate((np.random.normal(-1, 1, 50), np.random.normal(3, 1, 50)))
Y = np.concatenate((np.repeat(0., 50), np.repeat(1., 50)))
Xdata = tf.placeholder(shape=[1], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[1], dtype=tf.float32)

# Create variable (one model parameter = A)
A = tf.Variable(tf.random_normal(mean=10, shape=[1]))

# Add operation to graph
# Want to create the operstion sigmoid(x + A)
# Note, the sigmoid() part is in the loss function
res = tf.add(Xdata, A)

# Now we have to add another dimension to each (batch size of 1)
resExpanded = tf.expand_dims(res, 0)
YtargetExpanded = tf.expand_dims(Ytarget, 0)

# Add classification loss (cross entropy)
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=resExpanded, labels=YtargetExpanded)

# Create Optimizer
opt = tf.train.GradientDescentOptimizer(0.05)
step = opt.minimize(xentropy)

# Initialize
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Run loop
for i in range(1400):
    index = np.random.choice(100)
    x = [X[index]]
    y = [Y[index]]
    tf.reset_default_graph()
    sess.run(step, feed_dict = {Xdata: x, Ytarget: y})
    if (i+1)%200 == 0:
        print('Step : ' + str(i+1) + ' A = ' + str(sess.run(A)))
        print('Loss = ' + str(sess.run(xentropy, feed_dict={Xdata: x, Ytarget: y})))
        
# Evaluate Predictions
predictions = []
for i in range(len(X)):
    x_val = [X[i]]
    prediction = sess.run(tf.round(tf.sigmoid(res)), feed_dict={Xdata: x_val})
    predictions.append(prediction[0])
    
accuracy = sum(x==y for x,y in zip(predictions, Y))/100.
print('Final Accuracy = ' + str(np.round(accuracy, 2)))