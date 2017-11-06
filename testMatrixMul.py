import numpy as np
import tensorflow as tf

# list
a = [[1,2],[3,4]]
b = [[5,6],[7,8]]

# ndarray
a_array = np.array(a)
b_array = np.array(b)

# matrix
a_matrix = np.matrix(a)
b_matrix = np.matrix(b)

print("a:",a)
print("b:",b)
print("array *:")
print(a_array*b_array)
print("array np.dot():")
print(np.dot(a_array,b_array))
print("matrix np.multiply():")
print(np.multiply(a_matrix,b_matrix))
print("matrix *:")
print(a_matrix*b_matrix)

# tensorflow
a_tf = tf.Variable(a_array)
b_tf = tf.Variable(b_array)
multiply = tf.multiply(a_tf,b_tf)
matmul = tf.matmul(a_tf,b_tf)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print("tf tf.multiply():")
print(sess.run(multiply))
print("tf tf.matmul():")
print(sess.run(matmul))
