class Sequential:
    """
    A simple sequential container for layers.
    """

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def params_and_grads(self):
        for layer in self.layers:
            for name, param in layer.params.items():
                yield param, layer.grads[name]
