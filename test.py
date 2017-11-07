import tensorflow as tf
import numpy as np

a = tf.constant([1,2,3],tf.float32)
b = tf.reduce_mean(a)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print(sess.run(b))
