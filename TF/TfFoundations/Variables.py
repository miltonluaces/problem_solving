#Constants immutable values which do not change
#Placeholder assigned once and do not change after
#Variables are constanly recomputed

import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

#y = Wx + b
W = tf.Variable([2.5, 4.0], tf.float32, name='var_W')

x = tf.placeholder(tf.float32, name='x')
b = tf.Variable([5.0, 10.0], tf.float32, name='var_b')

y = W * x + b

#Initialize all variables defined
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    print("Final result: Wx + b = ", sess.run(y, feed_dict={x: [10, 100]}))

#s = W * x
s = W * x

#Initialize only those variables that you might need
init = tf.variables_initializer([W])

with tf.Session() as sess:
    sess.run(init)

    #print("Will this work?: Wx + b =", sess.run(y, feed_dict={x: [10, 100]}))
    print("Result: Wx =", sess.run(s, feed_dict={x: [10, 100]}))

number = tf.Variable(2)
multiplier = tf.Variable(1)

init = tf.global_variables_initializer()

result = number.assign(tf.multiply(number, multiplier))

with tf.Session() as sess:
    sess.run(init)

    for i in range(10):
        print("Result number * multiplier = ", sess.run(result))
        print("Increment multiplier, new value = ", sess.run(multiplier.assign_add(1)))

