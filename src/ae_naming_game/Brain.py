import tensorflow as tf
import os


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


def compute_loss_once(model, x):
    y = model.decode(model.encode(x), apply_sigmoid=True)
    return tf.reduce_mean(tf.square(tf.subtract(y, x))), y


def compute_apply_gradients_once(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss, _ = compute_loss_once(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


class Autoencoder(tf.keras.Model):

    def __init__(self, latent_dim=50):
        super(Autoencoder, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer((16, 16)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(latent_dim)
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer((latent_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(256),
            tf.keras.layers.Reshape((16, 16))
        ])

    def load(self):
        """ Load weights to models folder. """
        self.encoder.load_weights(os.path.join('data', 'models', 'encoder.h5'))
        self.decoder.load_weights(os.path.join('data', 'models', 'decoder.h5'))

    def save(self):
        """ Save weights to models folder. """
        self.encoder.save(os.path.join('data', 'models', 'encoder.h5'))
        self.decoder.save(os.path.join('data', 'models', 'decoder.h5'))

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
