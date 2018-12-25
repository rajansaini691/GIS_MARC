from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
import numpy as np

# Loads the dataset into separate variables for training
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizes data from the dataset
x_train, x_test = x_train / 255.0, x_test / 255.0

# Specifies the architecture of the neural network
model = Sequential()

# Adds a Convolution layer. It has 32 fields with 3 rows and columns each
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))

# Adds a pooling layer. This takes input from the convolution layer and puts it into a single neuron, reducing # parameters
model.add(MaxPooling2D(pool_size=(2,2)))

# The dropout layer drops out certain neurons, preventing overfitting
model.add(Dropout(0.25))

# Flattens convolutional layers so that they are compatible with fully-connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#model.load_weights("image_classifier.h5")

# Defines the loss function and optimizer of the network
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Adds the dataset to the model and starts training
model.fit(x_train, y_train, epochs=5, verbose=1, batch_size=32)

print(model.evaluate(x_test, y_test))

# Saves the model so we don't have to keep retraining
model.save("image_classifier.h5")

