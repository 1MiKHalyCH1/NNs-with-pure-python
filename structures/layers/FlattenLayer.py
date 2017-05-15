from structures.layers.AbstractLayer import AbstractLayer
import numpy as np


class FlattenLayer(AbstractLayer):
    def __init__(self, shape):
        self.shape = shape
        self.W = np.array([])

    def feed_forward(self, X):
        return np.concatenate(X).ravel()

    def backpropagate(self, delta, outputs, prev_W, result=None):
        return outputs.reshape(self.shape)

    def update(self, inputs, delta):
        pass