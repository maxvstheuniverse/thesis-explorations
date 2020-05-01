import numpy as np


def interpolate(a, b, nsteps):
    """ Interpolates two arrays in the given number of steps. """
    alpha = 1 / nsteps
    steps = []
    for i in range(1, nsteps + 1):
        alpha_star = alpha * i
        step = alpha_star * b + (1 - alpha_star) * a
        steps.append(step)

    return np.array(steps)


def text_to_int(text, char2idx):
    return np.array([char2idx[c] for c in text]).astype('uint64')


def texts_to_ints(texts, char2idx):
    return np.array([text_to_int(text, char2idx) for text in texts]).astype('uint64')


# -- RNN Helpers

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

