import gzip
import pickle
import numpy as np

import matplotlib.pyplot as plt


class DataManager:
    def __init__(self, filename):
        self.filename = 'mnist.pkl.gz'
        self.read_data()

    def draw_image(self, pictures, lables, size):
        for i in range(len(pictures)):
            ax = plt.subplot(*size, i + 1)
            ax.axis('off')
            ax.set_title(lables[i])
            plt.imshow(pictures[i], cmap='Greys', interpolation='None')
        plt.show()

    def read_data(self):
        with gzip.open(self.filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        # images = [e.reshape((28, 28)) for e in train_set[:, 0]]
        # labels = train_set[:, 1]
        # self.draw_image(images[:15], labels[:15], (3, 5))
        self.train_set, self.valid_set, self.test_set = [
            self.restruct(x) for x in data]

    def restruct(self, data):
        res = []
        for x, y in zip(*data):
            # x = x.reshape(28,28)
            y = np.array([1 if i == y else 0 for i in range(10)])
            res.append((x, y))
        return res

    def classify(self, x):
        return list(x).index(max(x))

if __name__ == '__main__':
    DataManger().read_data()