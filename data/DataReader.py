import gzip
import pickle
import numpy as np

import matplotlib.pyplot as plt


class DataReader:
    def __init__(self):
        self.read_data()

    def draw_image(self, pictures, lables, size):
        for i in range(len(pictures)):
            ax = plt.subplot(*size, i + 1)
            ax.axis('off')
            ax.set_title(lables[i])
            plt.imshow(pictures[i], cmap='Greys', interpolation='None')
        plt.show()

    def read_data(self):
        with gzip.open('mnist.pkl.gz', 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        train_set = np.array(list(zip(*train_set)))
        valid_set = np.array(list(zip(*valid_set)))
        test_set  = np.array(list(zip(*test_set)))
        # images = [e.reshape((28, 28)) for e in train_set[:, 0]]
        # labels = train_set[:, 1]
        # self.draw_image(images[:15], labels[:15], (3, 5))
        self.train_set, self.valid_set, self.test_set = train_set, valid_set, test_set
