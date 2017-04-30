import numpy as np

from structures.layers.AbstractLayer import AbstractLayer


class DenseLayer(AbstractLayer):
    def __init__(self, input_size, output_size, activation,
                 d_activation, bias=1.0, alpha=0.2):
        self.activation = activation
        self.d_activation = d_activation
        self.bias = np.array([bias]*output_size)
        self.W = np.random.randn(input_size, output_size) * 0.12 - 0.12
        self.res = None
        self.alpha = alpha

    def feed_forward(self, X):
        z = np.dot(self.W.T, X)+self.bias
        return self.activation(z)

    def backpropagation(self, inputs, delta):
        self.W += self.alpha * np.outer(inputs, delta)
        self.bias += self.alpha * delta
