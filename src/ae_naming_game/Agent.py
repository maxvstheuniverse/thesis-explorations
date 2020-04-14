import tensorflow as tf
import numpy as np
import random
import imageio
import os

from ae_naming_game.Brain import Autoencoder, compute_apply_gradients_once

tf.keras.backend.set_floatx('float64')

MIN_DISTANCE = .5


class Agent:
    """ Autoencoder Agent """

    def __init__(self, identifier, num_objects):
        self.optimizer = tf.keras.optimizers.Adam(1e-3)
        self.brain = Autoencoder()
        self.brain.load()

        self.identifier = identifier
        self.vocabulary = [[] for _ in range(num_objects)]

    def print_vocabulary(self, t):
        output_dir = os.path.join("output", "ae_vocabularies", f"{t}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, objects in enumerate(self.vocabulary):
            for j, word in enumerate(objects):
                inferred_word = np.array(self.brain.decode(word, apply_sigmoid=True)) * 255
                im = inferred_word.astype('uint8')
                imageio.imwrite(os.path.join(output_dir, f"a{self.identifier}_o{i}_j.jpg"), im)

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
