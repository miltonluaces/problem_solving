# Implementing Gates
#----------------------------------
#
# This function shows how to implement
# various gates in Tensorflow
#
# One gate will be one operation with
# a variable and a placeholder.
# We will ask Tensorflow to change the
# variable based on our loss function

import tensorflow as tf


# Start Graph Session
sess = tf.Session()

#----------------------------------
# Create a multiplication gate:
#   f(x) = a * x
#
#  a --
#      |
#      |---- (multiply) --> output
#  x --|
#

a = tf.Variable(tf.constant(4.))
x_val = 5.
Xdata = tf.placeholder(dtype=tf.float32)

multiplication = tf.multiply(a, Xdata)

# Declare the loss function as the difference between
# the output and a target value, 50.
loss = tf.square(tf.sub(multiplication, 50.))

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Declare optimizer
opt = tf.train.GradientDescentOptimizer(0.01)
step = opt.minimize(loss)

# Run loop across gate
print('Optimizing a Multiplication Gate Output to 50.')
for i in range(10):
    sess.run(step, feed_dict={Xdata: x_val})
    a_val = sess.run(a)
    mult_output = sess.run(multiplication, feed_dict={Xdata: x_val})
    print(str(a_val) + ' * ' + str(x_val) + ' = ' + str(mult_output))
    
#----------------------------------
# Create a nested gate:
#   f(x) = a * x + b
#
#  a --
#      |
#      |-- (multiply)--
#  x --|              |
#                     |-- (add) --> output
#                 b --|
#
#

# Start a New Graph Session
ops.reset_default_graph()
sess = tf.Session()

a = tf.Variable(tf.constant(1.))
b = tf.Variable(tf.constant(1.))
x_val = 5.
Xdata = tf.placeholder(dtype=tf.float32)

two_gate = tf.add(tf.multiply(a, Xdata), b)

# Declare the loss function as the difference between
# the output and a target value, 50.
loss = tf.square(tf.sub(two_gate, 50.))

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Declare optimizer
opt = tf.train.GradientDescentOptimizer(0.01)
step = opt.minimize(loss)

# Run loop across gate
print('\nOptimizing Two Gate Output to 50.')
for i in range(10):
    sess.run(step, feed_dict={Xdata: x_val})
    a_val, b_val = (sess.run(a), sess.run(b))
    two_gate_output = sess.run(two_gate, feed_dict={Xdata: x_val})
    print(str(a_val) + ' * ' + str(x_val) + ' + ' + str(b_val) + ' = ' + str(two_gate_output))