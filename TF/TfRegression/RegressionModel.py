import numpy as np
import tensorflow as tf
from sklearn import datasets, linear_model
from ReturnsData import read_goog_sp500_data

xData, yData = read_goog_sp500_data()

# Python benchmark implementation

fit = linear_model.LinearRegression()
fit.fit(xData.reshape(-1,1), yData.reshape(-1,1))
print(fit.coef_)
print(fit.intercept_)

# Tensorflow implementation

W = tf.Variable(tf.zeros([1,1]))
b = tf.Variable(tf.zeros([1]))

x = tf.placeholder(tf.float32, [None,1])
y = tf.matmul(x,W) + b
y_ = tf.placeholder(tf.float32, [None,1]) 
loss = tf.reduce_mean(tf.square(y_-y)) #MSE
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

epochs=10000
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(epochs):
    xs = np.array([[xData[i % len(xData)]]])
    ys = np.array([[yData[i % len(yData)]]])
    fd = {x:xs, y_:ys}
    sess.run(optimizer, feed_dict=fd)

    if (i+1)% 1000 == 0:
        print("After %d iteration: " % i)
        print("W: %f" % sess.run(W))
        print("b: %f" % sess.run(b))
        print("loss: %f" % sess.run(loss, feed_dict=fd))

