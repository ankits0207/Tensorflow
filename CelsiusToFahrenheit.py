import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np

import matplotlib.pyplot as plt

celsiusInput = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheitOutput = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# This layer would have 1 internal variable specified by units and takes 1d array as the input
layer0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# Assembling the defined layers(layer0 in this case) into model
model = tf.keras.Sequential([layer0])

# Compiling the model using 0.1 as the learning rate
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# Training the model with 500 epochs, 1 epoch = one complete representation of a data point to the model,
# 7 input-output pairs => 7*500 = 3500 times
log = model.fit(celsiusInput, fahrenheitOutput, epochs=500, verbose=False)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(log.history['loss'])
plt.show()

print(model.predict([100.0]))

print('Done')
