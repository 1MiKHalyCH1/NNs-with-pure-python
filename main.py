import numpy as np

from data.data_reader import DataReader
from structures.CNN import CNN
from structures.functions import sigmoid
from structures.layers.ConvolutionalLayer import ConvolutionalLayer
from structures.layers.FlattenLayer import FlattenLayer
from structures.layers.PoolingLayer import PoolingLayer

if __name__ == '__main__':
    dr = DataReader()
    a = dr.train_set
    X = [a[0][0].reshape((28, 28))]

    nn = CNN()
    nn.add_layer(ConvolutionalLayer(20, 5, 5, sigmoid))
    nn.add_layer(PoolingLayer((2, 2), np.max))
    nn.add_layer(ConvolutionalLayer(50, 5, 5, sigmoid))
    nn.add_layer(PoolingLayer((2, 2), np.max))
    nn.add_layer(FlattenLayer())

    res = nn.feed_forward(X)
    dr.draw_image(res[:100], (10,10))
