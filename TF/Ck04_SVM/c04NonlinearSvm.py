# Nonlinear SVM Example
#----------------------------------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

# Load the data (iris.data) = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
X = np.array([[x[0], x[3]] for x in iris.data])
Y = np.array([1 if y==0 else -1 for y in iris.target])
class1X = [x[0] for i,x in enumerate(X) if Y[i]==1]
class1Y = [x[1] for i,x in enumerate(X) if Y[i]==1]
class2X = [x[0] for i,x in enumerate(X) if Y[i]==-1]
class2Y = [x[1] for i,x in enumerate(X) if Y[i]==-1]

# Declare batch size
bSize = 150

# Initialize placeholders
Xdata = tf.placeholder(shape=[None, 2], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[None, 1], dtype=tf.float32)
predGrid = tf.placeholder(shape=[None, 2], dtype=tf.float32)

# Create variables for svm
b = tf.Variable(tf.random_normal(shape=[1,bSize]))

# Gaussian (RBF) kernel
gamma = tf.constant(-25.0)
dist = tf.reduce_sum(tf.square(Xdata), 1)
dist = tf.reshape(dist, [-1,1])
sq_dists = tf.add(tf.sub(dist, tf.multiply(2., tf.matmul(Xdata, tf.transpose(Xdata)))), tf.transpose(dist))
my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

# Compute SVM Model
res = tf.matmul(b, my_kernel)
first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
Ytarget_cross = tf.matmul(Ytarget, tf.transpose(Ytarget))
second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, Ytarget_cross)))
loss = tf.neg(tf.sub(first_term, second_term))

# Gaussian (RBF) prediction kernel
rA = tf.reshape(tf.reduce_sum(tf.square(Xdata), 1),[-1,1])
rB = tf.reshape(tf.reduce_sum(tf.square(predGrid), 1),[-1,1])
pred_sq_dist = tf.add(tf.sub(rA, tf.multiply(2., tf.matmul(Xdata, tf.transpose(predGrid)))), tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

prediction_output = tf.matmul(tf.multiply(tf.transpose(Ytarget),b), pred_kernel)
prediction = tf.sign(prediction_output-tf.reduce_mean(prediction_output))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(Ytarget)), tf.float32))

# Declare optimizer
opt = tf.train.GradientDescentOptimizer(0.01)
step = opt.minimize(loss)

# Initialize variables
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
lossVec = []
batch_accuracy = []
for i in range(300):
    index = np.random.choice(len(X), size=bSize)
    x = X[index]
    y = np.transpose([Y[index]])
    sess.run(step, feed_dict={Xdata: x, Ytarget: y})
    
    ls = sess.run(loss, feed_dict={Xdata: x, Ytarget: y})
    lossVec.append(ls)
    
    acc_temp = sess.run(accuracy, feed_dict={Xdata: x,
                                             Ytarget: y,
                                             predGrid:x})
    batch_accuracy.append(acc_temp)
    
    if (i+1)%75==0:
        print('Step : ' + str(i+1))
        print('Loss = ' + str(ls))

# Create a mesh to plot points in
xMin, xMax = X[:, 0].min() - 1, X[:, 0].max() + 1
yMin, yMax = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.02),
                     np.arange(yMin, yMax, 0.02))
grid_points = np.c_[xx.ravel(), yy.ravel()]
[grid_predictions] = sess.run(prediction, feed_dict={Xdata: x,
                                                   Ytarget: y,
                                                   predGrid: grid_points})
grid_predictions = grid_predictions.reshape(xx.shape)

# Plot points and grid
plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
plt.plot(class1X, class1Y, 'ro', label='I. setosa')
plt.plot(class2X, class2Y, 'kx', label='Non setosa')
plt.title('Gaussian SVM Results on Iris Data')
plt.xlabel('Pedal Length')
plt.ylabel('Sepal Width')
plt.legend(loc='lower right')
plt.ylim([-0.5, 3.0])
plt.xlim([3.5, 8.5])
plt.show()

# Plot batch accuracy
plt.plot(batch_accuracy, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss over time
plt.plot(lossVec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

# sess.run(prediction_output, feed_dict={Xdata: x, Ytarget: y, prediction_grid: grid_points})
# sess.run(pred_kernel, feed_dict={Xdata: x, Ytarget: y, prediction_grid: grid_points})
# sess.run(res, feed_dict={Xdata:x, Ytarget: y})
# sess.run(second_term, feed_dict={Xdata:x, Ytarget: y})
