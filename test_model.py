"""
Written by Jordan Otsuji

test_model.py tests the accuracy of the model(s) created by create_model.py
"""
import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load data again
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

modelPath25 = "digit_recognition_25_15_10.model"
modelPath64 = "digit_recognition_64_64_10.model"
modelPath128 = "digit_recognition_128_128_10.model"
model25 = tf.keras.models.load_model(modelPath25)
model64 = tf.keras.models.load_model(modelPath64)
model128 = tf.keras.models.load_model(modelPath128)


# flatten layer, 2 hidden layers, 1 output layer 
# [layer0, layer1, layer2, layer3] = model25.layers
# W1,b1 = layer1.get_weights()
# W2,b2 = layer2.get_weights()
# W3,b3 = layer3.get_weights()
# print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}")
# print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
# print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")

loss, accuracy = model25.evaluate(x_test, y_test)
print(f"{modelPath25} loss: {loss}")
print(f"{modelPath25} accuracy: {accuracy}")
loss, accuracy = model64.evaluate(x_test, y_test)
print(f"{modelPath64} loss: {loss}")
print(f"{modelPath64} accuracy: {accuracy}")
loss, accuracy = model128.evaluate(x_test, y_test)
print(f"{modelPath128} loss: {loss}")
print(f"{modelPath128} accuracy: {accuracy}")
