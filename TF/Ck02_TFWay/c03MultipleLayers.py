# Multiple layers
#----------------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import Functions as fn

# Intro: conv2d. Create image and kernel to convolute
kernelOrig = tf.constant([
    [1, 0, 1],
    [2, 1, 0],
    [0, 0, 1]
], dtype=tf.float32, name='k')

inputOrig = tf.constant([
    [4, 3, 1, 0],
    [2, 1, 0, 1],
    [1, 2, 4, 1],
    [3, 1, 0, 2]
], dtype=tf.float32, name='i')

# Reshape to tensorflow format: kernel [height, width, inChannel, outChannel], image: [num, height, width, channel]
kernel = tf.reshape(kernelOrig, [3, 3, 1, 1], name='kernel')
image  = tf.reshape(inputOrig, [1, 4, 4, 1], name='image')
res = tf.squeeze(tf.nn.conv2d(image, kernel, [1, 1, 1, 1], "VALID")) # VALID means no padding

with tf.Session() as sess:
   print(sess.run(res))


# Create a small random 'image' of size 4x4
xSh = [1, 4, 4, 1]
X = np.random.uniform(size=xSh)
Xp = tf.placeholder(tf.float32, shape=xSh)

# Spatial moving window average (2x2 with a stride of 2 for height and width). filter=0.25 average of the 2x2 window
filter = tf.constant(0.25, shape=[2, 2, 1, 1])
strides = [1, 2, 2, 1]
MALayer= tf.nn.conv2d(Xp, filter, strides, padding='SAME', name='Moving_Avg_Window')

# Define a custom layer which will be sigmoid(Ax+b) where x is a 2x2 matrix and A and b are 2x2 matrices
def CustomLayer(inMat):
    sqMat = tf.squeeze(inMat)
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape=[2, 2])
    Ax = tf.matmul(A, sqMat)
    Axb = tf.add(Ax, b)
    return(tf.sigmoid(Axb))

# Add custom layer in a scope
with tf.name_scope('Custom_Layer') as scope:
    cl1 = CustomLayer(MALayer)

# After custom operation, size is now 2x2 (squeezed out size 1 dims)
sess = tf.Session()
print(sess.run(cl1, feed_dict={Xp: X}))

fn.Tboard()