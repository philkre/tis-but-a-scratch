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
