from datetime import datetime
from data.DataManager import DataManager
from structures.CNN import CNN
from structures.functions import sigmoid, d_sigmoid, same, d_same
from structures.layers.DenseLayer import DenseLayer


def classify(arr):
    a, b = max((x, y) for y, x in enumerate(arr))
    return b


if __name__ == '__main__':
    dm = DataManager()

    nn = CNN()
    nn.add_layer(DenseLayer(784, 400, sigmoid, d_sigmoid))
    nn.add_layer(DenseLayer(400, 150, sigmoid, d_sigmoid))
    nn.add_layer(DenseLayer(150, 50, sigmoid, d_sigmoid))
    nn.add_layer(DenseLayer(50, 10, same, d_same))

    curtime = datetime.now()
    nn.train(dm.train_set, epochs=30, epoch_range=1)
    t = datetime.now() - curtime
    print('{} seconds for training'.format(t.seconds))
    print('correct for {}% on train set'.format(nn.calculate_correct(dm.test_set)))

    results = []
    for x, y in dm.test_set[:75]:
        res = nn.feed_forward(x)
        res = dm.classify(res)
        results.append((x.reshape((28, 28)), res))
    results = list(zip(*results))
    dm.draw_image(*results[:75], (5, 15))
