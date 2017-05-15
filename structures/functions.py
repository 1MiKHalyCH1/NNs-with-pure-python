import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - tanh(x) ** 2


def ReLU(x):
    return x * (x > 0)


def d_ReLU(x):
    return 1. * (x > 0)


def same(x):
    return x


def d_same(x):
    return np.ones(x.shape)
