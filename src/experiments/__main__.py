import tensorflow as tf

from tensorflow import keras

"""
The idea is to create

--> encoder --> latent --> decoder -->





--> embeddings --> rnn --> generator --.
"""


def main(args=None):
    (train_data, test_data), info = tfds.load('imdb_reviews/subwords8k',
                                              split=(tfds.Split.TRAIN, tfds.Split.TEST),
                                              with_info=True, as_supervised=True)

    embedding_layer = keras.layers.Embedding(1000, 5)
    result = embedding_layer(tf.constant([[0, 1, 2], [3, 4, 5]]))
    print(result.numpy())
    print(result.shape)
