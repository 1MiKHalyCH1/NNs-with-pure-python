import numpy as np

from structures.layers.AbstractLayer import AbstractLayer


class ConvolutionalLayer(AbstractLayer):
    def __init__(self, n_filters, filter_size, activation,
                 d_activation, alpha=0.02):
        self.width, self.height = filter_size
        self.W = np.array([np.random.rand(*filter_size) * 2 - 1
                           for _ in range(n_filters)])
        self.activation = activation
        self.d_activation = d_activation
        self.alpha = alpha

    def feed_forward(self, X):
        self.inputs = X
        self.outputs = [self.activation(self.convolve(f_map, w))
                        for f_map in X for w in self.W]
        return self.outputs

    def convolve(self, image, filter, mode='valid'):
        if mode == 'full':
            padding = [(filter.shape[0] - 1)] * 2
            image = np.pad(image, padding, 'constant')
        try:
            in_width, in_height = image.shape
        except Exception as e:
            print(e)
        f_width, f_height = filter.shape
        out_width = in_width - f_width + 1
        out_height = in_height - f_height + 1

        res = np.zeros((out_width, out_height))
        for x in range(out_width):
            for y in range(out_height):
                a = image[x:x + f_width, y:y + f_height].ravel()
                res[x][y] = filter.T.ravel().dot(a)
        return res

    def backpropagate(self, delta, outputs, prev_W, result=None):
        delta = [self.d_activation(x) for x in delta]
        res = []
        for i in range(len(self.inputs)):
            part = delta[i * len(self.W):(i + 1) * len(self.W)]
            a = [self.convolve(part[i], np.rot90(self.W[i], 2),'full')
                 for i in range(len(self.W))]
            res.append(np.mean(a, axis=0) * self.inputs[i])
        return res

    def update(self, inputs, delta):
        d_w = []
        for i in range(len(self.W)):
            part = self.outputs[i::len(self.W)]
            res = [self.convolve(delta[i], np.rot90(part[i], 2))
                   for i in range(len(part))]
            d_w.append(np.mean(res, axis=0))
        a = 0.002 * np.array(d_w)
        self.W += a
