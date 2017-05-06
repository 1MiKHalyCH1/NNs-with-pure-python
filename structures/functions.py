import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    res = np.tanh(x)
    if np.nan in res:
        print('nan')
    return res


def d_tanh(x):
    res = 1 - tanh(x) ** 2
    if np.nan in res:
        print('nan')
    return res


def LReLU(x):
    return np.maximum(x, 0.01, x)


def d_LReLU(x):
    return np.ones(x.shape)
