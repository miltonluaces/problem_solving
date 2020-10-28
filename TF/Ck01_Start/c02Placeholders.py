# Placeholders
#----------------------------------

import numpy as np
import tensorflow as tf

# Declaration
x = tf.placeholder(tf.float32, shape=(4, 4))
y = tf.identity(x)
rndArray = np.random.rand(4, 4)

print('\nCAP 2. PLACEHOLDERS\n')

sess = tf.Sessionsess = tf.Session()
print(sess.run(y, feed_dict={x: rndArray}))

print("\nExecution Ok")