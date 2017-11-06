# get MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnistData/", one_hot=True)

# # draw the MNIST Input Images
# import matplotlib.pyplot as plt
# trainImage = mnist.train.images[0,:]
# trainLabel = mnist.train.labels[0]
# testImage = mnist.test.images[0,:]
# testLabel = mnist.test.labels[0]
# validationImage = mnist.validation.images[0,:]
# validationLabel = mnist.validation.labels[0]
# plt.subplot(131);plt.imshow(trainImage.reshape(28,28),'gray_r')
# plt.subplot(132);plt.imshow(testImage.reshape(28,28),'gray_r')
# plt.subplot(133);plt.imshow(validationImage.reshape(28,28),'gray_r')
# print(trainLabel)
# print(testLabel)
# print(validationLabel)
# plt.show()









# LSTM NN
import tensorflow as tf
#define constants
time_steps = 28  #unrolled through 28 time steps
num_units = 128  #hidden LSTM units
n_input = 28  #rows of 28 pixels
learning_rate = 0.001  #learning rate for adam
n_classes = 10  #mnist is meant to be classified in 10 classes(0-9).
batch_size = 128  #size of batch

# 1 input
x = tf.placeholder("float",[None, time_steps, n_input])
y = tf.placeholder("float",[None, n_classes])
# processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input=tf.unstack(x ,time_steps,1)

# 2 draw the graph
# defining the network
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units,forget_bias=1)
# rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units)
# gru_cell = tf.contrib.rnn.GRUCell(num_units)
outputs,_ = tf.contrib.rnn.static_rnn(lstm_cell,input,dtype="float32")
# initial_state = lstm_cell.zero_state(batch_size, tf.float32)
# outputs, _states = tf.nn.dynamic_rnn(cnn_cell, input, initial_state=initial_state, dtype=tf.float32)
# converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
weights = tf.Variable(tf.random_normal([num_units,n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))
prediction = tf.matmul(outputs[-1],weights) + bias
#model evaluation
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 3 back-prop training
# loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#optimization
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

#4 init & sess
#initialize variables
init=tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#5 training
iter=1
while iter<800:
    batch_x,batch_y = mnist.train.next_batch(batch_size=batch_size)
    batch_x = batch_x.reshape((batch_size,time_steps,n_input))
    sess.run(train, feed_dict={x: batch_x, y: batch_y})
    if iter %10==0:
        acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
        los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
        print("For iter ",iter)
        print("Accuracy ",acc)
        print("Loss ",los)
        print("__________________")
    iter = iter+1



#calculating test accuracy
test_data = mnist.test.images[:128].reshape((-1, time_steps, n_input))
test_label = mnist.test.labels[:128]
print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
