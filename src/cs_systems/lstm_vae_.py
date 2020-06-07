import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps MNIST digits to a triplet (z_mean, z_log_var, z)."""

    def __init__(self, latent_dim=2, hidden_dim=64, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.rnn = keras.layers.LSTM(hidden_dim)
        self.dense_mean = keras.layers.Dense(latent_dim)
        self.dense_logvar = keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.rnn(inputs)

        z_mean = self.dense_mean(x)
        z_logvar = self.dense_logvar(x)
        z = self.sampling((z_mean, z_logvar))

        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, timesteps, original_dim, hidden_dim=64, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.repeat = keras.layers.RepeatVector(timesteps, input_shape=(2,))
        self.rnn = keras.layers.LSTM(hidden_dim)
        self.collector = keras.layers.TimeDistributed(keras.layers.Dense(original_dim, activation='softmax'))

    def call(self, inputs):
        x = self.repeat(inputs)
        x = self.rnn(x)
        return self.collector(x)


class RecurrentVariationalAutoEncoder(keras.Model):
    def __init__(self, timesteps, original_dim, hidden_dim, latent_dim, name="RNN-VAE", **kwargs):
        super(RecurrentVariationalAutoEncoder, self).__init__(name=name, **kwargs)

        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim, hidden_dim)
        self.decoder = Decoder(timesteps, original_dim, hidden_dim)

    def call(self, inputs):
        z_mean, z_logvar, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        kl_loss = -0.5 * tf.reduce_sum(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed
