import abc


class AbstractLayer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def feed_forward(self, X):
        pass

    @abc.abstractmethod
    def backpropagation(self):
        pass
