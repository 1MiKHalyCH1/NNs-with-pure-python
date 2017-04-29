import numpy as np

from structures.layers.AbstractLayer import AbstractLayer


class ConvolutionalLayer(AbstractLayer):
    def __init__(self, n_filters, filter_width, filter_height, activation):
        self.width, self.height = filter_width, filter_height
        self.W = [np.random.randn(filter_width, filter_height) for _ in range(n_filters)]
        self.activation = activation

    def feed_forward(self, X):
        output = []
        in_width, in_height = X[0].shape
        out_width = in_width - self.width + 1
        out_height = in_height - self.height + 1

        for f_map in X:
            for w in self.W:
                res = np.zeros((out_width, out_height))
                for x in range(out_width):
                    for y in range(out_height):
                        a = f_map[x:x + self.width,
                            y:y + self.height].ravel()
                        res[x][y] = w.T.ravel().dot(a)
                res = self.activation(res)
                output.append(res)
        return output
