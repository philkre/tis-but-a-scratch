class SGD:
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, model):
        for param, grad in model.params_and_grads():
            param -= self.lr * grad
