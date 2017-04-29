import numpy as np

from structures.layers.AbstractLayer import AbstractLayer


class PoolingLayer(AbstractLayer):
    def __init__(self, W_shape, activation):
        self.width, self.height = W_shape
        self.activation = activation

    def feed_forward(self, X):
        output = []
        in_width, in_height = X[0].shape
        out_width = in_width // 2
        out_height = in_height // 2
        for f_map in X:
            res = np.zeros((out_width, out_height))
            for x in range(out_width):
                for y in range(out_height):
                    f_part = f_map[x * self.width:(x + 1) * self.width,
                             y * self.height:(y + 1) * self.height]
                    res[x][y] = self.activation(f_part)
            output.append(res)
        return output
