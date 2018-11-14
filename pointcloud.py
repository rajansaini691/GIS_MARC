import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist

# Loads the dataset into separate variables for training
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# The images that need classification. Normalizes them. 
x_train, x_test = x_train / 255.0, x_test / 255.0

# Specifies the architecture of the neural network 
model = tf.eras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.ketras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
