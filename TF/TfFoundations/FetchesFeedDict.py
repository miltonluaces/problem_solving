import tensorflow as tf

# y = Wx + b
W = tf.constant([10, 100], name='const_W')

# Note that these placeholders can hold tensors of any shape
x = tf.placeholder(tf.int32, name='x')
b = tf.placeholder(tf.int32, name='b')

Wx = tf.multiply(W, x, name="Wx")

y = tf.add(Wx, b, name="y")

# y_ = x - b
y_ = tf.subtract(x, b, name="y_")


with tf.Session() as sess:
    print("Intermediate result: Wx = ", sess.run(Wx, feed_dict={x: [3, 33]}))

    print("Final result: Wx + b = ", sess.run(y, feed_dict={x: [5, 50], b: [7, 9]}))

    print("Intermediate specified: Wx + b = ", sess.run(fetches=y, feed_dict={Wx: [100, 1000], b: [7, 9]}))

    print("Two results: [Wx : b, x -  b] = ", sess.run(fetches=[y, y_], feed_dict={x: [5, 50], b: [7, 9]}))

writer = tf.summary.FileWriter('./tensorboards/m3_example2', sess.graph)
writer.close()
