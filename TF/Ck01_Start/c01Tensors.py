# Tensors
#----------------------------------

import tensorflow as tf
 
# Declare a tensor
rows = 2
cols = 3 
zeroTensor = tf.zeros([rows,cols])

# Variable declaration
zeros = tf.Variable(tf.zeros([rows, cols]))
ones = tf.Variable(tf.ones([rows, cols]))
zerosLike = tf.Variable(tf.zeros_like(zeros))
onesLike = tf.Variable(tf.ones_like(ones))
filled = tf.Variable(tf.fill([rows, cols], -1))
const = tf.Variable(tf.constant([8, 6, 7, 5, 3, 0, 9]))
constFilled = tf.Variable(tf.constant(-1, shape=[rows, cols]))
normFilled = tf.random_normal([rows, cols], mean=0.0, stddev=1.0)

# Sequence generation
seqLin = tf.Variable(tf.linspace(start=0.0, stop=1.0, num=3)) # Generates [0.0, 0.5, 1.0] includes the end (equal to np.linspace)
seqRng = tf.Variable(tf.range(start=6, limit=15, delta=3)) # Generates [6, 9, 12] doesn't include the end (similar to range)

# Run
print('\nCAP 1. TENSORS')
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    z = sess.run(zeros); print("zeros = \n", z)
    o = sess.run(ones); print("ones = \n", o)
    zl = sess.run(zerosLike); print("zerosLike = \n", zl)
    ol = sess.run(onesLike); print("onesLike = \n", ol)
    f = sess.run(filled); print("filled = \n", f)
    c = sess.run(const); print("const = \n", c)
    cf = sess.run(constFilled); print("constFilled = \n", cf)
    nf = sess.run(normFilled); print("normFilled = \n", nf)
    sl = sess.run(seqLin); print("seqLin = \n", sl)
    sr = sess.run(seqRng); print("seqRng = \n", sr)



# Tensorboard:
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

print("\nExecution Ok")