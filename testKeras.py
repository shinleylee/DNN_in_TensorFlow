import keras

# state a model
model = keras.models.Sequential()
#1 draw the graph
model.add(keras.layers.Dense(units=64, input_dim=100))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Dense(units=10))
model.add(keras.layers.Activation("softmax"))

#2 compile
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['Accuracy'])
# model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=0.01,momentum=0.9,nesterov=True))

#3 train/fit
hist = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
# validation_split : split last 20% input as validation set
# note that the input should be shuffled manually before, because keras will process validation_split before auto_shuffle
# model.train_on_batch(x_batch, y_batch)
print(hist.history) # hist.history includes the loss and other metrics after each epoch

#4 predict on test
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
classes = model.predict(x_test, batch_size=128)







# save and load the Keras model (in HDF5 format)
model.save(filepath)
del model # deletes the existing model
model = keras.models.load_model(filepath)
