# Multi-class (Nonlinear) SVM Example
#------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets


# Create graph
sess = tf.Session()

# Load the data (iris.data) = [(Sepal Length, Sepal Width, Petal Length, Petal Width)].  Y: (I. setosa, I. virginica, I. versicolor) (3 classes to do a one vs all classification).
iris = datasets.load_iris()
X = np.array([[x[0], x[3]] for x in iris.data])
Y1 = np.array([1 if y==0 else -1 for y in iris.target])
Y2 = np.array([1 if y==1 else -1 for y in iris.target])
Y3 = np.array([1 if y==2 else -1 for y in iris.target])
Y = np.array([Y1, Y2, Y3])
class1X = [x[0] for i,x in enumerate(X) if iris.target[i]==0]
class1Y = [x[1] for i,x in enumerate(X) if iris.target[i]==0]
class2X = [x[0] for i,x in enumerate(X) if iris.target[i]==1]
class2Y = [x[1] for i,x in enumerate(X) if iris.target[i]==1]
class3X = [x[0] for i,x in enumerate(X) if iris.target[i]==2]
class3Y = [x[1] for i,x in enumerate(X) if iris.target[i]==2]

# Declare batch size
bSize = 50

# Initialize placeholders
Xdata = tf.placeholder(shape=[None, 2], dtype=tf.float32)
Ytarget = tf.placeholder(shape=[3, None], dtype=tf.float32)
predGrid = tf.placeholder(shape=[None, 2], dtype=tf.float32)

# Create variables for svm
b = tf.Variable(tf.random_normal(shape=[3,bSize]))

# Gaussian (RBF) kernel
gamma = tf.constant(-10.0)
dist = tf.reduce_sum(tf.square(Xdata), 1)
dist = tf.reshape(dist, [-1,1])
sq_dists = tf.add(tf.sub(dist, tf.multiply(2., tf.matmul(Xdata, tf.transpose(Xdata)))), tf.transpose(dist))
kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

# Declare function to do reshape/batch multiplication
def reshape_matmul(mat):
    v1 = tf.expand_dims(mat, 1)
    v2 = tf.reshape(v1, [3, bSize, 1])
    return(tf.batch_matmul(v2, v1))

# Compute SVM Model
res = tf.matmul(b, kernel)
first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
Ytarget_cross = reshape_matmul(Ytarget)

second_term = tf.reduce_sum(tf.multiply(kernel, tf.multiply(b_vec_cross, Ytarget_cross)),[1,2])
loss = tf.reduce_sum(tf.neg(tf.sub(first_term, second_term)))

# Gaussian (RBF) prediction kernel. The prediction of a point will be the category with the largest margin or distance to boundary.
rA = tf.reshape(tf.reduce_sum(tf.square(Xdata), 1),[-1,1])
rB = tf.reshape(tf.reduce_sum(tf.square(predGrid), 1),[-1,1])
pred_sq_dist = tf.add(tf.sub(rA, tf.multiply(2., tf.matmul(Xdata, tf.transpose(predGrid)))), tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

prediction_output = tf.matmul(tf.multiply(Ytarget,b), pred_kernel)
prediction = tf.arg_max(prediction_output-tf.expand_dims(tf.reduce_mean(prediction_output,1), 1), 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(Ytarget,0)), tf.float32))

# Declare optimizer
opt = tf.train.GradientDescentOptimizer(0.01)
step = opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
lossVec = []
batchAccuracy = []
for i in range(100):
    index = np.random.choice(len(X), size=bSize)
    x = X[index]
    y = Y[:,index]
    sess.run(step, feed_dict={Xdata: x, Ytarget: y})
    
    ls = sess.run(loss, feed_dict={Xdata: x, Ytarget: y})
    lossVec.append(ls)
    
    acc_temp = sess.run(accuracy, feed_dict={Xdata: x,
                                             Ytarget: y,
                                             predGrid:x})
    batchAccuracy.append(acc_temp)
    
    if (i+1)%25 == 0:
        print('Step : ' + str(i+1))
        print('Loss = ' + str(ls))

# Create a mesh to plot points in
xMin, xMax = X[:, 0].min() - 1, X[:, 0].max() + 1
yMin, yMax = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(xMin, xMax, 0.02), np.arange(yMin, yMax, 0.02))
gridPoints = np.c_[xx.ravel(), yy.ravel()]
gridPreds = sess.run(prediction, feed_dict={Xdata: x, Ytarget: y, predGrid: gridPoints})
gridPreds = gridPreds.reshape(xx.shape)

# Plot points and grid
plt.contourf(xx, yy, gridPreds, cmap=plt.cm.Paired, alpha=0.8)
plt.plot(class1X, class1Y, 'ro', label='I. setosa')
plt.plot(class2X, class2Y, 'kx', label='I. versicolor')
plt.plot(class3X, class3Y, 'gv', label='I. virginica')
plt.title('Gaussian SVM Results on Iris Data')
plt.xlabel('Pedal Length')
plt.ylabel('Sepal Width')
plt.legend(loc='lower right')
plt.ylim([-0.5, 3.0])
plt.xlim([3.5, 8.5])
plt.show()

# Plot batch accuracy
plt.plot(batchAccuracy, 'k-', label='Accuracy')
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