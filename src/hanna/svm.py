import numpy as np

from sklearn import svm
from functools import partial


def hanna_gaussian_kernel(x, mu, sigma):
    """ Hanna's Kernel non-linear Gaussian Function.

        φ(x,μ)=exp[ –||x–μ||^2 /σ^2 ]
    """
    return np.exp(-np.sum((x - mu) ** 2) / sigma ** 2)


def proxy_kernel(X, Y, K, sigma=5):
    """ (Pre-)calculates Gram Matrix K. """
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))

    for i, x in enumerate(X):
        x = x.flatten()
        for j, y in enumerate(Y):
            y = y.flatten()
            gram_matrix[i, j] = K(x, y, sigma)

    return gram_matrix


def SVM(sigma=5):
    """ Returns the fitted support vector machine using hanna's gaussian kernel. """
    return svm.SVC(kernel=partial(proxy_kernel, K=hanna_gaussian_kernel, sigma=sigma))
