from structures.layers.AbstractLayer import AbstractLayer
import numpy as np


class FlattenLayer(AbstractLayer):
    def feed_forward(self, X):
        return np.concatenate(X).ravel()
