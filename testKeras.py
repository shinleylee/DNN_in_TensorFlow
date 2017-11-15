import keras

# Sequential Model

# state a model
model = keras.models.Sequential()
#1 draw the graph
model.add(keras.layers.Dense(units=64, input_dim=100))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Dense(units=10))
model.add(keras.layers.Activation("softmax"))

# 2 compile
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['Accuracy'])
# model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=0.01,momentum=0.9,nesterov=True))

# 3 train/fit
hist = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
# validation_split : split last 20% input as validation set
# note that the input should be shuffled manually before, because keras will process validation_split before auto_shuffle
# model.train_on_batch(x_batch, y_batch)
print(hist.history) # hist.history includes the loss and other metrics after each epoch

# 4 predict on test
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
classes = model.predict(x_test, batch_size=128)







# save and load the Keras model (in HDF5 format)
model.save(filepath)
del model # deletes the existing model
model = keras.models.load_model(filepath)







# Functional Model

# 0 define inputs
inputs = keras.layers.Input(shape=(784,), dtype='int32', name='main_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

# 1 draw the graph
x = keras.layers.Embedding(output_dim=512, input_dim=10000, input_length=100)(inputs)
lstm_out = keras.layers.LSTM(32)(x)
x = keras.layers.Dense(output_dim=64,activation='relu')(lstm_out)
x = keras.layers.Dense(output_dim=64,activation='relu')(x)
predictions = keras.layers.Dense(output_dim=10,activation='softmax')(x)

model = keras.models.Model(input=inputs,output=predictions)
model = keras.models.Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

# 2 compile
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 3 train
model.fit(data, labels)  # starts training
