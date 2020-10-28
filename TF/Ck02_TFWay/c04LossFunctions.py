# Loss Functions
#----------------------------------

import matplotlib.pyplot as plt
import tensorflow as tf


# REGRESSION

# Variables
sess = tf.Session()
X = tf.linspace(-1., 1., 500)
Y = tf.constant(0.)
d1 = tf.constant(0.25)
d2 = tf.constant(5.)

# 1. L2 loss: L = (pred - actual)^2
l2 = tf.square(Y - X)
l2Out = sess.run(l2)

# 2. L1 loss: L = abs(pred - actual)
l1 = tf.abs(Y - X)
l1Out = sess.run(l1)

# 3. Pseudo-Huber loss: L = delta^2 * (sqrt(1 + ((pred-exp)/delta)^2) - 1)
ph1 = tf.multiply(tf.square(d1), tf.sqrt(1. + tf.square((Y - X)/d1)) - 1.)
ph1Out = sess.run(ph1)

ph2 = tf.multiply(tf.square(d2), tf.sqrt(1. + tf.square((Y - X)/d2)) - 1.)
ph2Out = sess.run(ph2)

# Plot
Xarr = sess.run(X)
plt.plot(Xarr, l2Out, 'b-', label='L2 Loss')
plt.plot(Xarr, l1Out, 'r--', label='L1 Loss')
plt.plot(Xarr, ph1Out, 'k-.', label='P-Huber Loss (0.25)')
plt.plot(Xarr, ph2Out, 'g:', label='P-Huber Loss (5.0)')
plt.ylim(-0.2, 0.4)
plt.legend(loc='lower right', prop={'size': 11})
plt.show()


# CLASSIFICATION

# Variables
X = tf.linspace(-3., 5., 500)
Y = tf.constant(1.)
targets = tf.fill([500,], 1.)
weight = tf.constant(0.5)
unscaledLogits = tf.constant([[1., -3., 10.]])
targetDist = tf.constant([[0.1, 0.02, 0.88]])
sparseTargetDist = tf.constant([2])

# 1. Hinge loss: L = max(0, 1 - (pred * exp)). Use for predicting binary (-1, 1) classes
hi = tf.maximum(0., 1. - tf.multiply(Y, X))
hiOut = sess.run(hi)
#tf.nn.

# 2. Cross entropy loss: L = -exp * (log(pred)) - (1-exp)(log(1-pred))
ce = - tf.multiply(Y, tf.log(X)) - tf.multiply((1. - Y), tf.log(1. - X))
ceOut = sess.run(ce)

# 3. Sigmoid entropy loss: L = -exp * (log(sigmoid(pred))) - (1-exp)(log(1-sigmoid(pred)))
se = tf.nn.sigmoid_cross_entropy_with_logits(logits=X, labels=targets)
seOut = sess.run(se)

# 4. Weighted (softmax) cross entropy loss: L = -exp * (log(pred)) * weights - (1-exp)(log(1-pred))
wce = tf.nn.weighted_cross_entropy_with_logits(X, targets, weight)
wceOut = sess.run(wce)

# 5. Softmax entropy loss : L = -exp * (log(softmax(pred))) - (1-exp)(log(1-softmax(pred)))
sme = tf.nn.softmax_cross_entropy_with_logits(logits=unscaledLogits, labels=targetDist)
print(sess.run(sme))

# 6. Sparse entropy loss : L = sum(-exp * log(pred)). Use when classes and targets have to be mutually exclusive
spe = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=unscaledLogits, labels=sparseTargetDist)
print(sess.run(spe))

# Plot 
Xarr = sess.run(X)
plt.plot(Xarr, hiOut, 'b-', label='Hinge Loss')
plt.plot(Xarr, ceOut, 'r--', label='Cross Entropy Loss')
plt.plot(Xarr, ceOut, 'k-.', label='Cross Entropy Sigmoid Loss')
plt.plot(Xarr, wceOut, 'g:', label='Weighted Cross Entropy Loss (x0.5)')
plt.ylim(-1.5, 3)
#plt.xlim(-1, 3)
plt.legend(loc='lower right', prop={'size': 11})
plt.show()