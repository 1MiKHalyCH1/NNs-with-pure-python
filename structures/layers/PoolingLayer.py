import numpy as np
from itertools import product

from structures.layers.AbstractLayer import AbstractLayer


class PoolingLayer(AbstractLayer):
    def __init__(self, W_shape=(2, 2)):
        self.w_width, self.w_height = W_shape
        self.W = None

    def feed_forward(self, X):
        self.inputs = X
        output = []
        in_width, in_height = X[0].shape
        w_range = list(range(0, in_width + 1, self.w_width))
        h_range = list(range(0, in_height + 1, self.w_height))
        w_indexes = list(zip(w_range, w_range[1:]))
        h_indexes = list(zip(h_range, h_range[1:]))

        for f_map in X:
            res = np.array([np.max(f_map[x1:x2, y1:y2])
                            for (x1, x2), (y1, y2) in
                            product(w_indexes, h_indexes)])
            res = res.reshape(len(w_indexes), len(h_indexes))
            output.append(res)
        return output

    def backpropagate(self, delta, outputs, prev_W, result=None):
        width, height = outputs[0].shape
        res = [np.asarray(
            [[d[i // self.w_width, j // self.w_height]
              for j in range(width * self.w_width)]
             for i in range(height * self.w_height)])
            for d in delta]
        return res

    def update(self, inputs, delta):
        pass