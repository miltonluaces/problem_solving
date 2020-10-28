# Activation Functions
#----------------------------------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sess = tf.Session()

# X range
x = np.linspace(start=-10., stop=10., num=100)

# Y activations
yRelu = tf.nn.relu(x).eval()
yRelu6 = tf.nn.relu6(x).eval()
ySigmoid = tf.nn.sigmoid(x).eval()
yTanh = tf.nn.tanh(x).eval()
ySoftsign = tf.nn.softsign(x).eval()
ySoftplus = tf.nn.softplus(x).eval()
yElu = tf.nn.elu(x).eval()

# Plot the different functions
print('\nCAP 5. ACTIVATION FUNCTIONS')
plt.plot(x, ySoftplus, 'r--', label='Softplus', linewidth=2)
plt.plot(x, yRelu, 'b:', label='ReLU', linewidth=2)
plt.plot(x, yRelu6, 'g-.', label='ReLU6', linewidth=2)
plt.plot(x, yElu, 'k-', label='ExpLU', linewidth=0.5)
plt.ylim([-1.5,7])
plt.legend(loc='top left')
plt.show()

plt.plot(x, ySigmoid, 'r--', label='Sigmoid', linewidth=2)
plt.plot(x, yTanh, 'b:', label='Tanh', linewidth=2)
plt.plot(x, ySoftsign, 'g-.', label='Softsign', linewidth=2)
plt.ylim([-2,2])
plt.legend(loc='top left')
plt.show()
