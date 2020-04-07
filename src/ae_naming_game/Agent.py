import tensorflow as tf
import numpy as np
import random

from ae_naming_game.Brain import Autoencoder, compute_apply_gradients_once

MIN_DISTANCE = 1
tf.keras.backend.set_floatx('float64')


class Agent:
    """ Autoencoder Agent """

    def __init__(self, identifier, num_objects):
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.brain = Autoencoder()
        self.brain.load()

        self.identifier = identifier
        self.vocabulary = [[] for _ in range(num_objects)]

    def speak(self, obj):
        num_words = len(self.vocabulary[obj])

        if num_words != 0:
            latent_vars, invented = self.vocabulary[obj][random.randint(0, num_words - 1)], False
        else:
            latent_vars, invented = self._invent_word(), True

        if invented:
            self.vocabulary[obj].append(latent_vars)

        return self.brain.decode(latent_vars, apply_sigmoid=True), invented

    def listen(self, obj, word):
        inferred_word = self.brain.decode(self.brain.encode(word), apply_sigmoid=True)
        # compute_apply_gradients_once(self.brain, inferred_word, self.optimizer)
        distance = np.linalg.norm(word - inferred_word)
        return obj if distance < MIN_DISTANCE else -1

    def adopt(self, obj, word):
        latent_vars = self.brain.encode(word)
        self.vocabulary[obj] = [latent_vars]

    def add_word(self, obj, word):
        self.vocabulary[obj].append(self.brain.encode(word))

    def _invent_word(self):
        return np.array(tf.random.normal(shape=(1, self.brain.latent_dim))).astype('float64')
        # return self.brain.sample()[0]
