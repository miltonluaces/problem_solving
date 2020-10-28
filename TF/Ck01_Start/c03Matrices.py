# Matrices and Matrix Operations
#----------------------------------

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import h5py
import numpy as np
import tensorflow as tf
import Functions as fn
import warnings


# Declaring matrices
I = tf.diag([1.0,1.0,1.0], name="Identity_Matrix")
A = tf.truncated_normal([2,3], name="Normal")
B = tf.fill([2,3], 5.0, name="Constant")
C = tf.random_uniform([3,2], name="Uniform")
D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]), name="From_np_array")

# Basic operations
S = A+B
R = B-B
M = tf.matmul(B, I, name="Matrix_product")

# Matrix operations
T = tf.transpose(C, name="Transp")
E = tf.matrix_determinant(D, name="Det")
In = tf.matrix_inverse(D, name="Inverse")
Ch =tf.cholesky(I, name="Cholesky")
EE = tf.self_adjoint_eig(D)

# Run
print('\nCAP 3. MATRICES')
fn.Run([I, A, B, C, D, S, R, M, T, E, In, Ch]) 
fn.runStr(EE, "Eigenvals and vecs:\n")