# Operations
#----------------------------------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import Functions as fn

# div() vs truediv() vs floordiv()
D1 = tf.div(3,4, name='int_div')
D2 = tf.truediv(3,4, name='float_div')
D3 = tf.floordiv(3.0,4.0, name='floor_of_float_div')

# Mod function
M = tf.mod(22.0,5.0, name='module')

# Cross Product
Cp = tf.cross([1.,0.,0.],[0.,1.,0.], name='cross_product')
#region Cross Product
# Example: C = (2,3,4) x (5,6,7) + (-3, 6, -3) where  
#   cx = aybz − azby = 3×7 − 4×6 = −3
#   cy = azbx − axbz = 4×5 − 2×7 = 6
#   cz = axby − aybx = 2×6 − 3×5 = −3 
#endregion

# Trigonometrics functions
S = tf.sin(3.1416, name='sin')
C = tf.cos(3.1416, name='cos')
# Tangemt
Tg = tf.div(tf.sin(3.1416/4.), tf.cos(3.1416/4.), name='tg') 

# Custom operation
nums = range(15)
#Eq = tf.equal(nums, 3)

def poly(x):
    return(tf.subtract(3 * tf.square(x), x) + 10) # 3x^2 - x + 10

P = poly(11)

# What should we get with list comprehension
exp = [3*x*x-x+10 for x in nums]
print(exp)

# Tensorflow custom function output
for num in nums:
    poly(num)

print('\nCAP 4. OPERATIONS')
# Run
fn.Run([D1, D2, D3, M, Cp, S, C, Tg]) 
