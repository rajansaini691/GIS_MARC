import skimage
import tensorflow as tf
import numpy as np
from PIL import Image

# Loads mnist dataset (set of handwrittend digits)
mnist = tf.keras.datasets.mnist

# Sets formatting options when printing to console
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)

# Loads an example image, normalizes it, and flattens it
img = Image.open('8.png').convert('L')
arr = np.array(img) / 255.0
arr = arr.ravel() 

# Puts the example image into a numpy array
ex = np.array(img)

# Loads the dataset into separate variables for training
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizes data from the dataset
x_train, x_test = x_train / 255.0, x_test / 255.0

# Specifies the architecture of the neural network 
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(784,)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.load_weights("image_classifier.h5")

# Defines the loss function and optimizer of the network
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Adds the dataset to the model and starts training
#model.fit(x_train, y_train, epochs=3)

print(model.evaluate(x_test, y_test))

# Saves the model so we don't have to keep retraining
#model.save_weights("image_classifier.h5")

# Prints the model's prediction of the example image
print(model.predict(np.array([arr]), verbose=1))
print(x_train[0])
