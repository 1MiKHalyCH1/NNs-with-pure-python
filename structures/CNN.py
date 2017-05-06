from random import choice, random, randint

import numpy as np


class CNN:
    def __init__(self):
        self.layers = []
        self.inputs = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def get_result(self):
        return self.layers[-1].result

    def feed_forward(self, X):
        self.inputs.clear()
        for layer in self.layers:
            self.inputs.append(X)
            X = layer.feed_forward(X)
        return X

    def backpropagation(self, Y, res):
        error = Y - res
        d = self.layers[-1].d_activation(res)
        deltas = [np.multiply(error, d)]
        for i in range(len(self.layers) - 2, -1, -1):
            dot = np.dot(deltas[-1], self.layers[i + 1].W.T)
            d = self.layers[i].d_activation(self.inputs[i + 1])
            deltas.append(np.multiply(dot, d))
        deltas.reverse()
        for i in range(len(self.layers)):
            layer = self.layers[i]
            input = self.inputs[i]
            delta = deltas[i]
            layer.backpropagation(input, delta)

    def train(self, data, epochs=10000, epoch_range=100):
        for i in range(1, epochs+1):
            for x, y in data:
                res = self.feed_forward(x)
                self.backpropagation(y, res)
            if not i % epoch_range:
                res = np.array([(self.feed_forward(x) - y) for x, y in data])
                correct = (1 - (1/len(data)) * np.sum(res ** 2)) * 100
                print('epoch:{}, correct for {:.0f}%'.format(i, correct))
