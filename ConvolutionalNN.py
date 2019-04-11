import tensorflow as tf
import tensorflow_datasets as tfds
tf.logging.set_verbosity(tf.logging.ERROR)

import math
import numpy as np
import matplotlib.pyplot as plt

import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

# Each image: 28 pixels height and 28 pixels width, thus every image is represented as 28*28=784 pixels

# tf.enable_eager_execution()


def normalize(data, label):
    data = tf.cast(data, tf.float32)
    data /= 255
    return data, label


# Load the data
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train, test = dataset['train'], dataset['test']

classLabels = ['Top', 'Lower', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

numTrain = metadata.splits['train'].num_examples
numTest = metadata.splits['test'].num_examples
print(numTrain)
print(numTest)

# Preprocess the data, each pixel is an integer belonging to the range [0, 255]
train = train.map(normalize)
test = test.map(normalize)

# # Visualize the data
# for image, label in test.take(1):
#     break
# image = image.numpy().reshape((28, 28))
# plt.figure()
# plt.imshow(image, cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show()

# Setting up the model with 3 layers:
# First convoluted layer is added with 32 convolutional outputs, 3*3 filter,
# 0 padding all around over the 28*28*1 3d input
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D((2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ]
)

# Setting the compilation parameters for the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batchSize = 32
train = train.repeat().shuffle(numTrain).batch(batchSize)
test = test.batch(batchSize)
model.fit(train, epochs=5, steps_per_epoch=math.ceil(numTrain/batchSize))

test_loss, test_accuracy = model.evaluate(test, steps=math.ceil(numTest/batchSize))
print(test_accuracy)