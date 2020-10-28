# Operations on a Computational Graph
#------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Create data to feed in
X = np.array([1., 3., 5., 7., 9.])
Xp = tf.placeholder(tf.float32)
c = tf.constant(3.)

# Product
sess = tf.Session()
pr = tf.multiply(Xp, c)
for x in X:
    print(sess.run(pr, feed_dict={Xp : x}))

