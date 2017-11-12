import keras

# state a model
model = keras.models.Sequential()

# draw the graph
model.add(keras.layers.Dense(units=64, input_dim=100))
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Dense(units=10))
model.add(keras.layers.Activation("softmax"))

# compile
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['Accuracy'])
# model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=0.01,momentum=0.9,nesterov=True))

# train/fit
model.fit(x_train, y_train, epochs=5, batch_size=32)
# model.train_on_batch(x_batch, y_batch)

# predict on test
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
classes = model.predict(x_test, batch_size=128)
