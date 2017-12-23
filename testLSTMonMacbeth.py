# This is a LSTM Demo on Text Sequence
# Data: Macbeth https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/12/10165151/macbeth.txt
# Implement: Keras
# Reference: https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw%3D%3D&mid=2650734862&idx=1&sn=1a2adda4da7bd7509f10556e8ae218f4#wechat_redirect

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils


# load file
filename = "macbeth.txt"
filepath = "Text Sequence/"
text = (open(filepath + filename).read()).lower()


# mapping characters with integers
unique_chars = sorted(list(set(text)))
char_to_int = {}
int_to_char = {}
for i, c in enumerate(unique_chars):
    char_to_int.update({c: i})
    int_to_char.update({i: c})


# preparing input and output dataset
X = []
Y = []
for i in range(0, len(text)-50, 1):
    sequence = text[i:i+50]
    label = text[i+50]
    X.append([char_to_int[char] for char in sequence])
    Y.append(char_to_int[label])


# reshaping, normalizing and one hot encoding
X_modified = numpy.reshape(X, (len(X), 50, 1))  # input form [sample, time_step, feature]
X_modified = X_modified / float(len(unique_chars))  # normalizing
Y_modified = np_utils.to_categorical(Y)


# defining the LSTM model
model = Sequential()
model.add(LSTM(300, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(300))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# fitting the model
model.fit(X_modified, Y_modified, epochs=1, batch_size=30)

# picking a random seed
start_index = numpy.random.randint(0, len(X)-1)
new_string = X[start_index]

# generating characters
for i in range(50):
    x = numpy.reshape(new_string, (1, len(new_string), 1))
    x = x / float(len(unique_chars))

    # predict
    pred_index = numpy.argmax(model.predict(x, verbose=0))
    char_out = int_to_char[pred_index]
    seq_in = [int_to_char[value] for value in new_string]
    print(char_out)

    new_string.append(pred_index)
    new_string = new_string[1: len(new_string)]
