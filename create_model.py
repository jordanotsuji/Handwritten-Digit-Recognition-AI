"""
Written by Jordan Otsuji

create_model.py trains the model and saves it for later use
"""
import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# MNIST dataset contains 28x28 pixel labeled hand written digits 
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
# flatten layer to change 28x28 input to a 1 dimensional input
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# Each layer of the neural network, relu activation function (linear when positive, 0 if otherwise)
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
# softmax activation function for probability distribution output, with highest # as the model's classification
model.add(tf.keras.layers.Dense(10, activation="softmax"))


model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
        )

# train model
model.fit(x_train, y_train, epochs=5)


# save model
model.save("digit_recognition_128_128_10.model")
