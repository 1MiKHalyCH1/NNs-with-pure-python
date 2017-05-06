import gzip
import pickle

import matplotlib.pyplot as plt


class DataReader:
    def __init__(self):
        self.read_data()

    def draw_image(self, pictures, size):
        for i in range(len(pictures)):
            ax = plt.subplot(*size, i + 1)
            ax.axis('off')
            plt.imshow(pictures[i], cmap='Greys', interpolation='None')
        plt.show()

    def read_data(self):
        with gzip.open('mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        # images = [e.reshape((28, 28)) for e in train_set[0]]
        # draw_image(images[:1], (1, 1))
        train_set = list(zip(*train_set))
        self.train_set, self.valid_set, self.test_set = train_set, valid_set, test_set
