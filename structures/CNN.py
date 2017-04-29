class CNN:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def feed_forward(self, X):
        for layer in self.layers:
            X = layer.feed_forward(X)
        return X
