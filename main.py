import numpy as np

from data.DataReader import DataReader
from structures.CNN import CNN
from structures.functions import tanh, d_tanh, sigmoid, d_sigmoid
from structures.layers.DenseLayer import DenseLayer


def classify(arr):
    a, b = max((x, y) for y, x in enumerate(arr))
    return b


if __name__ == '__main__':
    dr = DataReader()
    data = np.array(dr.train_set)[:400]
    X, y = list(zip(*data))
    Y = []
    for i in range(len(y)):
        answer = y[i]
        r = np.zeros(10)
        r[answer] = 1
        Y.append(r)
    data = list(zip(X, Y))

    # nn = CNN()
    # nn.add_layer(ConvolutionalLayer(20, 5, 5, sigmoid))
    # nn.add_layer(PoolingLayer((2, 2), np.max))
    # nn.add_layer(ConvolutionalLayer(50, 5, 5, sigmoid))
    # nn.add_layer(PoolingLayer((2, 2), np.max))
    # nn.add_layer(FlattenLayer())

    # res = nn.feed_forward(X)
    # dr.draw_image(res[:100], (10,10))

    nn = CNN()
    nn.add_layer(DenseLayer(784, 500, sigmoid, d_sigmoid))
    nn.add_layer(DenseLayer(500, 50, sigmoid, d_sigmoid))
    nn.add_layer(DenseLayer(50, 10, lambda x: x, lambda x: np.ones(x.shape)))

    nn.train(data, 150, epoch_range=10)

    results = []
    for x, y in dr.test_set[:15]:
        res = nn.feed_forward(x)
        res = classify(res)
        results.append((x.reshape((28,28)), res))
    results = list(zip(*results))
    dr.draw_image(*results, (3,5))

def classify(x:list):
    return x.index(max(x))
