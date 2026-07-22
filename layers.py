import numpy as np


class Dense:
    """
    Creates a fully connected layer with forward and backward methods.
    """

    def __init__(self, in_feats: int, out_feats: int, seed=1004):
        """
        Init self.params and self.grads with He initialization.
        """
        rng = np.random.default_rng(seed)

        self.params = {
            "W": rng.normal(0, np.sqrt(2 / in_feats), (in_feats, out_feats)),
            "b": np.zeros((1, out_feats)),
        }

        self.grads = {
            "W": np.zeros_like(self.params["W"]),
            "b": np.zeros_like(self.params["b"]),
        }

        self.x = None

    def forward(self, x) -> np.ndarray:
        """
        x: (N, in_feats) -> returns (N, out_feats)
        """
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout) -> np.ndarray:
        """
        dout: (N, out_feats)
        returns: (N, in_feats)
        sets: self.grads["W"] (in_feats, out_feats), self.grads["b"] (1, out_feats)
        """
        self.grads["W"] = self.x.T @ dout
        self.grads["b"] = np.sum(dout, axis=0, keepdims=True)
        return dout @ self.params["W"].T


class ReLU:
    """
    Creates a ReLU activation layer with forward and backward methods.
    """

    def __init__(self):
        self.grads = {}
        self.params = {}
        self.x = None

    def forward(self, x) -> np.ndarray:
        """
        x: (N, D) -> returns (N, D)
        """
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout) -> np.ndarray:
        """
        dout: (N, D)
        returns: (N, D)
        """
        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx


class SoftmaxCrossEntropy:
    """
    Creates a Softmax Cross-Entropy loss layer with forward and backward methods.
    """

    def __init__(self):
        self.grads = {}
        self.params = {}
        self.x = None
        self.y = None
        self.prop = None

    def forward(self, x, y) -> np.ndarray:
        """
        x: (N, D) - logits
        y: (N, D) - one-hot labels
        returns: scalar loss
        """
        self.x = x
        self.y = y

        # compute softmax
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.prob = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # compute cross-entropy loss
        N = x.shape[0]
        loss = -np.sum(y * np.log(self.prob + 1e-15)) / N
        return loss

    def backward(self):
        N = self.x.shape[0]
        dx = (self.prob - self.y) / N
        return dx


class Flatten:
    """
    Creates a Flatten layer with forward and backward methods.
    """

    def __init__(self):
        self.grads = {}
        self.params = {}
        self.x_shape = None

    def forward(self, x) -> np.ndarray:
        """
        x: (N, C, H, W) -> returns (N, C*H*W)
        """
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout) -> np.ndarray:
        """
        dout: (N, C*H*W)
        returns: (N, C, H, W)
        """
        return dout.reshape(self.x_shape)
