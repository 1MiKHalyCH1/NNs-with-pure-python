import numpy as np

from structures.CNN import CNN
from structures.functions import tanh, d_tanh
from structures.layers.DenseLayer import DenseLayer


def classify(arr):
    a, b = max((x, y) for y, x in enumerate(arr))
    return b


if __name__ == '__main__':
    # dr = DataReader()
    # X, y = dr.train_set
    # X, y = X[:1000], y[:1000]
    # Y = []
    # for i in range(len(y)):
    #     answer = y[i]
    #     r = np.zeros(10)
    #     r[answer] = 1
    #     Y.append(r)
    # data = X, Y

    # nn = CNN()
    # nn.add_layer(ConvolutionalLayer(20, 5, 5, sigmoid))
    # nn.add_layer(PoolingLayer((2, 2), np.max))
    # nn.add_layer(ConvolutionalLayer(50, 5, 5, sigmoid))
    # nn.add_layer(PoolingLayer((2, 2), np.max))
    # nn.add_layer(FlattenLayer())

    # res = nn.feed_forward(X)
    # dr.draw_image(res[:100], (10,10))

    nn = CNN()
    nn.add_layer(DenseLayer(2, 4, tanh, d_tanh))
    nn.add_layer(DenseLayer(4, 4, tanh, d_tanh))
    nn.add_layer(
        DenseLayer(4, 1, lambda x: x, lambda x: np.ones(x.shape)))

    data = [
        [[0, 0],
         [0, 1],
         [1, 0],
         [1, 1]],
        [0, 1, 1, 0]
    ]
    nn.train(data)

    X, y = data
    for i in range(len(X)):
        res = nn.feed_forward(X[i])
        print('predicted:{}, right answer:{}'.format(res, y[i]))
