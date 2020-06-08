import numpy as np
import argparse
import random
from tensorflow import keras

from cs_systems.lstm_vae import VariationalRecurrentAutoEncoder


def create_ctable(characters):
    def encode(sequence):
        encoded_s = []
        for sc in sequence:
            enc = [1 if sc == c else 0 for i, c in enumerate(characters)]
            encoded_s.append(enc)

        return encoded_s

    def decode(sequence, calc_argmax=True):
        if calc_argmax:
            sequence = sequence.argmax(axis=-1)

        return [characters[i] for i in sequence]

    return encode, decode


def vectorize(data, encode):
    return np.array([encode(sequence) for sequence in data], dtype='float32')


def generate_data(size, timesteps, reverse=True):
    samples = []
    samples_reverse = []
    # -- one octave + equal amount of rests. 50/50 chance of rest?
    characters = [str(i) for i in range(24, 36)] + [' ' for _ in range(24, 36)]

    while len(samples) < size:
        notes = [random.randint(0, len(characters) - 1) for _ in range(timesteps)]
        tune = [characters[i] for i in notes]

        # if reverse:
        #     tune = tune[::-1]

        samples.append(tune)
        samples_reverse.append(tune[::-1])

    return samples, samples_reverse


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agents", type=int, default=4, dest="agents",
                        help="The number of agents in the simulation. Default: 4")
    args = parser.parse_args()

    characters = [' '] + [str(i) for i in range(24, 36)]
    encode, decode = create_ctable(characters)

    timesteps = 10
    hidden_dim = 128
    latent_dim = 30

    data, data_reverse = generate_data(50000, timesteps)
    x = vectorize(data, encode)
    x_reverse = vectorize(data_reverse, encode)

    print(x[0])

    batch_size = 64

    vae = VariationalRecurrentAutoEncoder(timesteps, len(characters), hidden_dim, latent_dim)
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    vae.compile(optimizer, loss='categorical_crossentropy')
    vae.fit(x_reverse, x, epochs=25, batch_size=128)

    vae.summary()
    vae.save('rnn_vae.tf')

    vae.encode(x_reverse)

    y = vae.predict(x_reverse[:1])
    print(y[0])
    print(decode(y[0]))
    print(decode(x[0]))

