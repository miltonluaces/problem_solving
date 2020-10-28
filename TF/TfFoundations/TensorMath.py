import tensorflow as tf

x = tf.constant([100, 200, 300], name='x')
y = tf.constant([1, 2, 3], name='y')

sum_x = tf.reduce_sum(x, name="sum_x")
prod_y = tf.reduce_prod(y, name="prod_y")

final_div = tf.div(sum_x, prod_y)

final_mean = tf.reduce_mean([sum_x, prod_y], name="final_mean")

sess = tf.Session()

print ("x: ", sess.run(x))
print ("y: ", sess.run(y))

print ("sum(x): ", sess.run(sum_x))
print ("prod(y): ", sess.run(prod_y))
print ("sum(x) / prod(y): ", sess.run(final_div))
print ("mean(sum(x), prod(y)): ", sess.run(final_mean))

writer = tf.summary.FileWriter('./tensorboards/m2_example4', sess.graph)

writer.close()
sess.close()




