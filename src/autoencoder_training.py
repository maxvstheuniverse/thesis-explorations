import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import imageio
import time
import math

from ae_naming_game.Brain import Autoencoder, compute_apply_gradients, compute_loss

tf.keras.backend.set_floatx('float64')


def main():
    # -- for some reason, black lines on white bg appears to work better.
    # -- Hence the -1. The original data has white lines on a black bg.

    x_ims = 1 - np.load(os.path.join('data', 'sets', 'line_language.npy')).astype('float64') / 255.0
    print(x_ims.shape)
    # y_ims = x_ims.flatten().reshape(x.shape[0], 256)

    latent_dim = 50
    batch_size = 4
    buffer = x_ims.shape[0]
    num_batches = math.floor(buffer / batch_size)

    x_train = tf.data.Dataset.from_tensor_slices(x_ims).shuffle(buffer).batch(batch_size)

    model = Autoencoder(latent_dim)
    model.summarize()

    model.load()

    optimizer = tf.keras.optimizers.Adam(1e-3)
    epochs = 5

    for epoch in range(epochs):
        start_time = time.time()
        for i, x in enumerate(x_train):
            print(f"Training samples. Batch {i} of {num_batches - 1}", end="\r")
            compute_apply_gradients(model, x, optimizer)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        acc = tf.keras.metrics.Accuracy()

        for x in x_train:
            mse, y = compute_loss(model, x)
            loss.update_state(mse)
            acc.update_state(x, y)

        print(f"Epoch {epoch:03d}: Loss {loss.result():0.05f}, Accuracy {acc.result():0.05f}, Time {(end_time - start_time):0.03f}s")

    model.save()

    # imageio.imwrite('output/sample.jpg', np.array(model.sample(None)[0] * 255).astype('uint8'))
    #     pd.DataFrame(history.history).plot(figsize=(8, 5))
    #     plt.grid(True)
    #     plt.gca().set_ylim(0, 1)
    #     plt.savefig(os.path.join('output', 'autoencoder.svg'))

    # -- Export results

    predictions = model.decode(model.encode(x_ims[:16]), apply_sigmoid=True)
    fig, axs = plt.subplots(16, 3, figsize=(12, 32))

    for i, ax in enumerate(axs):
        actual, prediction = x_ims[i], predictions[i]

        print(f"Distance Sample {i:02d}: {np.linalg.norm(actual - prediction)}")

        if i == 0:
            ax[0].set_title("Actual")
            ax[1].set_title("Predicted")
            ax[2].set_title("Difference")

        ax[0].imshow(actual, cmap="gray")
        ax[1].imshow(prediction, cmap="gray")
        ax[2].imshow(actual - prediction, cmap="gray")  # difference, if the images are black it is a perfect match.

        plt.setp(ax[0].get_xticklabels(), visible=False)
        plt.setp(ax[0].get_yticklabels(), visible=False)
        ax[0].tick_params(axis='both', which='both', length=0)

        plt.setp(ax[1].get_xticklabels(), visible=False)
        plt.setp(ax[1].get_yticklabels(), visible=False)
        ax[1].tick_params(axis='both', which='both', length=0)

        plt.setp(ax[2].get_xticklabels(), visible=False)
        plt.setp(ax[2].get_yticklabels(), visible=False)
        ax[2].tick_params(axis='both', which='both', length=0)

    plt.savefig(os.path.join("output", "comparison_new.png"))


if __name__ == '__main__':
    main()
