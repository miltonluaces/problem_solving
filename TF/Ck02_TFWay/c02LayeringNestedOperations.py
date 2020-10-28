# Layering Nested Operations
#---------------------------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import Functions as fn

# Create data
arr = np.array([[1., 3., 5., 7., 9.], [-2., 0., 2., 4., 6.], [-6., -3., 0., 3., 6.]])
X = np.array([arr, arr+1]); print(X)
Xp = tf.placeholder(tf.float32, shape=(3, 5), name='Xp')
p1 = tf.constant([[1.],[0.],[-1.],[2.],[4.]], name='p1')
p2 = tf.constant([[2.]], name='p2')
a1 = tf.constant([[10.]], name='a1')

# 1st Operation Layer = Multiplication
pr1 = tf.matmul(Xp, p1, name='pr1')

# 2nd Operation Layer = Multiplication
pr2 = tf.matmul(pr1, p2, name='pr2')

# 3rd Operation Layer = Addition
add1 = tf.add(pr2, a1, name='add1')

sess = tf.Session()
for x in X:
    print(sess.run(add1, feed_dict={Xp: x}))

# Tensorboard:
fn.Tboard()
