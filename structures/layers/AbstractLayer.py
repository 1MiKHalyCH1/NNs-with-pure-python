import abc


class AbstractLayer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def feed_forward(self, X):
        pass

    @abc.abstractmethod
    def backpropagate(self, delta, outputs, prev_W, result=None):
        pass

    @abc.abstractmethod
    def update(self, inputs, delta):
        pass
