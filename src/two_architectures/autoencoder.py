import tensorflow as tf

from time import time
from os import path

tf.config.experimental_run_functions_eagerly(True)
tf.keras.backend.set_floatx('float64')


@tf.function
def compute_loss(model, x):
    y = model.decode(model.encode(x), apply_sigmoid=True)
    return tf.reduce_mean(tf.square(tf.subtract(y, x))), y


@tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss, _ = compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


class Autoencoder(tf.keras.Model):

    def __init__(self, latent_dim=8, _optimizer=None):
        super(Autoencoder, self).__init__()

        self.latent_dim = latent_dim
        self.optimizer = _optimizer if _optimizer is not None else tf.keras.optimizers.Adam(1e-3)

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer((16, 16)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(latent_dim, activation="sigmoid")
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer((latent_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Reshape((16, 16))
        ])

    def load(self):
        """ Load weights to models folder. """
        self.encoder.load_weights(path.join('data', 'models', 'encoder.h5'))
        self.decoder.load_weights(path.join('data', 'models', 'decoder.h5'))

    def save(self):
        """ Save weights to models folder. """
        self.encoder.save(path.join('data', 'models', 'encoder.h5'))
        self.decoder.save(path.join('data', 'models', 'decoder.h5'))

    def sample(self, z=None):
        """ Randomly sample from the Autoencoder """
        if z is None:
            z = tf.random.normal(shape=(1, self.latent_dim))
        return self.decode(z, apply_sigmoid=True)

    def encode(self, x):
        """ Return the latent variables for input x. """
        return self.encoder(x)

    def decode(self, z, apply_sigmoid=False):
        """ Returns the reconstruction using the given latent_vars """
        logits = self.decoder(z)
        if apply_sigmoid:
            return tf.sigmoid(logits)
        return logits

    def summarize(self):
        self.encoder.summary()
        self.decoder.summary()

    def train(self, data, epochs, verbose=1, batch_size=32):
        buffer = data.shape[0]
        num_batches = buffer // batch_size

        x_train = tf.data.Dataset.from_tensor_slices(data).shuffle(buffer).batch(batch_size)

        total_time = 0
        for epoch in range(epochs):
            start_time = time()
            for i, x in enumerate(x_train):
                # print(f"Training samples. Batch {i} of {num_batches - 1}", end="\r")
                compute_apply_gradients(self, x, self.optimizer)
            end_time = time()

            loss = tf.keras.metrics.Mean()
            acc = tf.keras.metrics.Accuracy()

            for x in x_train:
                mse, y = compute_loss(self, x)
                loss.update_state(mse)
                acc.update_state(x, y)

            time_elapsed = end_time - start_time
            total_time += time_elapsed
            if verbose == 1:
                print(f"Epoch {epoch:04d}: Loss {loss.result():0.05f}, Time {(time_elapsed):0.03f}s", end="\r")

        print(f"Trained {epochs:04d} epochs in {total_time:0.03f}s")


