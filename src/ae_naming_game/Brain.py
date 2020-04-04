import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


x = np.load(os.path.join('data', 'sets', 'line_language.npy')) / 255.0
y = x.flatten().reshape(1000, 256)

latent_dim = 32
batch_size = 4

autoencoder = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(16, 16)),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(latent_dim),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(256, activation="sigmoid")
])

autoencoder.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
autoencoder.summary()
history = autoencoder.fit(x, y, epochs=250, batch_size=batch_size)
autoencoder.save(os.path.join('data', 'models', 'autoencoder.h5'))

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.savefig(os.path.join('output', 'autoencoder.svg'))
