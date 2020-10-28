# -*- coding: utf-8 -*-
# Using Tensorboard
#----------------------------------
#
# We illustrate the various ways to use
#  Tensorboard

import os
import io
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Initialize a graph session
sess = tf.Session()

# Create a visualizer object
summary_writer = tf.train.SummaryWriter('tensorboard', tf.get_default_graph())

# Create tensorboard folder if not exists
if not os.path.exists('tensorboard'):
    os.makedirs('tensorboard')
print('Running a slowed down linear regression. '
      'Run the command: $tensorboard --logdir="tensorboard"  '
      ' Then navigate to http://127.0.0.0:6006')

# You can also specify a port option with --port 6006

# Wait a few seconds for user to run tensorboard commands
time.sleep(3)

# Some parameters
bSize = 50
generations = 100

# Create sample input data
Xdata = np.arange(1000)/10.
true_slope = 2.
y_data = Xdata * true_slope + np.random.normal(loc=0.0, scale=25, size=1000)

# Split into train/test
train_ix = np.random.choice(len(Xdata), size=int(len(Xdata)*0.9), replace=False)
test_ix = np.setdiff1d(np.arange(1000), train_ix)
Xdata_train, y_data_train = Xdata[train_ix], y_data[train_ix]
Xdata_test, y_data_test = Xdata[test_ix], y_data[test_ix]

# Declare placeholders
x_graph_input = tf.placeholder(tf.float32, [None])
y_graph_input = tf.placeholder(tf.float32, [None])

# Declare model variables
m = tf.Variable(tf.random_normal([1], dtype=tf.float32), name='Slope')

# Declare model
output = tf.multiply(m, x_graph_input, name='Batch_Multiplication')

# Declare loss function (L1)
residuals = output - y_graph_input
l2_loss = tf.reduce_mean(tf.abs(residuals), name="L2_Loss")

# Declare optimization function
optim = tf.train.GradientDescentOptimizer(0.01)
step = optim.minimize(l2_loss)

# Visualize a scalar
with tf.name_scope('Slope_Estimate'):
    tf.scalar_summary('Slope_Estimate', tf.squeeze(m))
    
# Visualize a histogram (errors)
with tf.name_scope('Loss_and_Residuals'):
    tf.histogram_summary('Histogram_Errors', l2_loss)
    tf.histogram_summary('Histogram_Residuals', residuals)



# Declare summary merging operation
summary_op = tf.merge_all_summaries()

# Initialize Variables
init = tf.global_variables_initializer()
sess.run(init)

for i in range(generations):
    batch_indices = np.random.choice(len(Xdata_train), size=bSize)
    x_batch = Xdata_train[batch_indices]
    y_batch = y_data_train[batch_indices]
    _, trainLoss, summary = sess.run([step, l2_loss, summary_op],
                             feed_dict={x_graph_input: x_batch,
                                        y_graph_input: y_batch})
    
    testLoss, test_resids = sess.run([l2_loss, residuals], feed_dict={x_graph_input: Xdata_test,
                                                                       y_graph_input: y_data_test})
    
    if (i+1)%10==0:
        print('Generation {} of {}. Train Loss: {:.3}, Test Loss: {:.3}.'.format(i+1, generations, trainLoss, testLoss))

    log_writer = tf.train.SummaryWriter('tensorboard')
    log_writer.add_summary(summary, i)
    time.sleep(0.5)

#Create a function to save a protobuf bytes version of the graph
def gen_linear_plot(slope):
    linear_prediction = Xdata * slope
    plt.plot(Xdata, y_data, 'b.', label='data')
    plt.plot(Xdata, linear_prediction, 'r-', linewidth=3, label='predicted line')
    plt.legend(loc='upper left')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return(buf)

# Add image to tensorboard (plot the linear fit!)
slope = sess.run(m)
plot_buf = gen_linear_plot(slope[0])
# Convert PNG buffer to TF image
image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
# Add the batch dimension
image = tf.expand_dims(image, 0)
# Add image summary
image_summary_op = tf.image_summary("Linear Plot", image)
image_summary = sess.run(image_summary_op)
log_writer.add_summary(image_summary, i)
log_writer.close()