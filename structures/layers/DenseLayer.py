import numpy as np

from structures.layers.AbstractLayer import AbstractLayer


class DenseLayer(AbstractLayer):
    def __init__(self, input_size, output_size, activation,
                 d_activation, bias=1.0, alpha=0.02):
        self.activation = activation
        self.d_activation = d_activation
        self.bias = np.array([bias] * output_size)
        self.W = np.random.rand(input_size, output_size) * 2 - 1
        self.res = None
        self.alpha = alpha

    def feed_forward(self, X):
        return self.activation(np.dot(self.W.T, X) + self.bias)

    def backpropagate(self, delta, outputs, prev_W, result=None):
        if result is not None:
            dot = result - outputs
        else:
            dot = np.dot(delta, prev_W.T)
        d = self.d_activation(outputs)
        return np.multiply(dot, d)

    def update(self, inputs, delta):
        self.W += self.alpha * np.outer(inputs, delta)
        self.bias += self.alpha * delta

