import tensorflow as tf

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
lm = W * x + b

y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(lm - y))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)

train = optimizer.minimize(loss)

# training data
Xtrain = [1, 2, 3, 4]
Ytrain = [0, -1, -2, -3]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(1000):
        sess.run(train, {x: Xtrain, y: Ytrain})

    # evaluate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: Xtrain, y: Ytrain})

    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


