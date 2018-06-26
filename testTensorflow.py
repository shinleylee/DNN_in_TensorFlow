import numpy as np
import tensorflow as tf

# 0 read data
x_data = np.float32(np.random.rand(100, 2))
y_data = np.dot(x_data, np.array([[0.1], [0.2]])) + 0.3


def addLayer(inputs, inputSize, outputSize, activateFunction=None):
    # w = tf.Variable(tf.random_normal([inputSize,outputSize]))
    w = tf.Variable(tf.random_uniform([inputSize, outputSize], -1, 1))
    # b = tf.Variable(tf.zeros([1,outputSize]) + 0.1)
    b = tf.Variable(tf.zeros([1, outputSize]) + 0.1)
    z = tf.add(tf.matmul(inputs, w), b)
    # Dropout Layer: keep_prob will be feed in sess.run
    # z = tf.nn.dropout(z, keep_prob)
    if activateFunction is None:
        a = z
    else:
        a = activateFunction(z)
    return a


# 1 define placeholder for input and groundtruth
x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])

# 2 draw the graph
y_ = addLayer(inputs=x, inputSize=2, outputSize=1)
# w = tf.Variable(tf.random_uniform([1,2],-1,1))
# b = tf.Variable(tf.zeros([1]))
# y = tf.matmul(w, x_data) + b

# 3 back-propagation training
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_-y), reduction_indices=[1]))
# loss = tf.reduce_mean(tf.square(tf.subtract(y_,y)))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 4 init & sess
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 5 for training
for step in range(0, 201):
    sess.run(train, feed_dict={x: x_data, y: y_data})
    if step % 20 == 0:
        print(step, sess.run(loss, feed_dict={x: x_data, y: y_data}))
