import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


def sampling(args):
    z_mean, z_log_sigma = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.)
    return z_mean + K.exp(z_log_sigma) * epsilon


class RecurrentVariationalAutoEncoder(tf.keras.Model):
    def __init__(self, timesteps, original_dim, hidden_dim, latent_dim, name="RVAE", **kwargs):
        super(RecurrentVariationalAutoEncoder, self).__init__(name=name, **kwargs)

        RNN = keras.layers.LSTM
        self.original_dim = original_dim

        # -- Encoder
        inputs = keras.layers.Input(shape=(timesteps, original_dim,))

        # z = RNN(100, return_sequences=True)(inputs)
        z = RNN(hidden_dim)(inputs)

        # -- encoder outputs
        z_mean = keras.layers.Dense(latent_dim)(z)
        z_logvar = keras.layers.Dense(latent_dim)(z)
        z = keras.layers.Lambda(sampling)([z_mean, z_logvar])

        self.encoder = keras.Model(inputs=[inputs], outputs=[z_mean, z_logvar, z], name="Encoder")

        # -- Decoder
        d_inputs = keras.layers.Input(shape=(latent_dim,))

        x = keras.layers.RepeatVector(timesteps)(d_inputs)
        # x = RNN(30, return_sequences=True)(x)
        x = RNN(hidden_dim, return_sequences=True)(x)

        # -- decoder outputs
        x = keras.layers.TimeDistributed(keras.layers.Dense(original_dim, activation="softmax"))(x)

        self.decoder = keras.Model(inputs=[d_inputs], outputs=[x], name="Decoder")

    def call(self, x):
        z_mean, z_logvar, z = self.encoder(x)
        reconstructed = self.decoder(z)

        kl_loss = -0.5 * tf.reduce_mean(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar))
        self.add_loss(kl_loss)
        return reconstructed

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

