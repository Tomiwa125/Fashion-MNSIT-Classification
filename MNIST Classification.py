import numpy as np
import tensorflow as tf
import keras

from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
# Load dataset
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()

trainX = trainX.reshape((trainX.shape[0], 28, 28, 1)).astype('float32') / 255.0
testX = testX.reshape((testX.shape[0], 28, 28, 1)).astype('float32') / 255.0

trainY = to_categorical(trainY)
testY = to_categorical(testY)

# Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax')) # Output layer for 10 classes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(trainX, trainY, epochs=10, batch_size=200, validation_data=(testX, testY))
# Make predictions on two images from test set
predictions = model.predict(testX[:2])

# Print predicted classes for the first two test images
predicted_classes = np.argmax(predictions, axis=1)
print(predicted_classes)