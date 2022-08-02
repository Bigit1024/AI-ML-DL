# creating a newral network

import tensorflow as tf
import numpy as np
from tensorflow import keras

# Printing stuff
import matplotlib.pyplot as plt

# Load a pre-defined dataset (70k of 28x28)
fashion_mnist = keras.datasets.fashion_mnist

# Pull out data from dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# by default returns 60k images for training, and then remaining 10k images here for testing


# Show data
# print(train_labels[0])
# print(train_images[0])
# plt.imshow(train_images[], cmap='gray', vmin=0, vmax=255)
# plt.show()


# Define our newral net structure
model = keras.Sequential([
    
    # First layer
    # imput is a 28x28 image ("Flatten" flattens the 28x28 into a single 784x1 imput layer)
    keras.layers.Flatten(input_shape=(28,28)),

    # hidden layer is 128 deep. relu returns the value, or 0 (works good enough. much faster) 
    keras.layers.Dense(units=128, activation=tf.nn.relu),
    # activation function acts in like a filtering mechanism within each layer, that can further filter out our data

    # output is 0-10 (depending on what piece of clothing it is). return maximum
    keras.layers.Dense(10, activation=tf.nn.softmax)
    # output layers is 10 nodes, going up and down and each of the node corresponds to one of the numbers
    # Dense means every node in each column is connected to every other node in each column
    # crossing over lines for connection
    ])

# Compile our model
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# loss function tells, how correct or how incorrect we are......
# if we are super wrong , thats where optimizer function comes in for optimization of our result to get close to perfection
# by automatically changing the value, after checking the loss accordingly and simultaneously

# train our model, using our training data
model.fit(train_images, train_labels, epochs=5)

# Test our model, using our testing data
test_loss = model.evaluate(test_images, test_labels)

plt.imshow(test_images[1], cmap='gray', vmin=0, vmax=255)
plt.show()

print(test_labels[1])

# Make Predictions
predictions = model.predict(test_images)

print(predictions [1])


# print out prediction
print(list(predictions[1]).index(max(predictions[1])))
