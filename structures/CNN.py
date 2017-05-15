import numpy as np
from scipy.spatial.distance import cdist
from random import shuffle


class CNN:
    def __init__(self):
        np.random.seed(123)
        self.layers = []
        self.inputs = []
        self.outputs = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def get_result(self):
        return self.layers[-1].result

    def feed_forward(self, X):
        self.inputs.clear()
        for layer in self.layers:
            self.inputs.append(X)
            X = layer.feed_forward(X)
            self.outputs.append(X)
        return X

    def backpropagate(self, Y):
        deltas = [self.layers[-1].backpropagate(None, self.outputs[-1], None, Y)]
        for i in range(len(self.layers) - 2, -1, -1):
            delta = deltas[-1]
            out = self.outputs[i]
            prev_W = self.layers[i + 1].W
            deltas.append(self.layers[i].backpropagate(delta, out, prev_W))
        deltas.reverse()

        for i in range(len(self.layers)):
            layer = self.layers[i]
            inputs = self.inputs[i]
            delta = deltas[i]
            layer.update(inputs, delta)

    def train(self, data, epochs=10000, epoch_range=100):
        print('Training starts')
        indexes = list(range(len(data)))
        for i in range(1, epochs+1):
            shuffle(indexes)
            for j in indexes:
                x, y = data[j]
                self.feed_forward(x)
                self.backpropagate(y)
            if not i % epoch_range:
                correct = self.calculate_correct(data)
                print('epoch:{}, correct for {:.2f}%'.format(i, correct))

    def calculate_correct(self, data):
        distances = [cdist(
            self.feed_forward(x)[np.newaxis],
            y[np.newaxis],
            metric='euclidean')[0][0] ** 2
                     for x, y in data]
        E = (1 / (2 * len(data))) * sum(distances)
        return (1 - E) * 100
