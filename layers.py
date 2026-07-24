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


class MaxPool2D:
    """
    Creates a maxpool layer in two dimensions with forward and backward methods.
    """

    def __init__(self, pool_size: int = 2, stride: int = 2):
        self.pool_size = pool_size
        self.stride = stride
        self.grads = {}
        self.params = {}
        self.x_shape = None
        self.argmax_cache = None

    def forward(self, x) -> np.ndarray:
        """
        x: (N, C, H, W) -> out: (N, C, H_out, W_out)
        H_out = (H - pool_size) // stride + 1
        W_out = (W - pool_size) // stride + 1
        Cache argmax index per window
        """
        self.x_shape = x.shape
        N, C, H, W = x.shape
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1

        out = np.zeros((N, C, H_out, W_out))
        self.argmax_cache = np.zeros((N, C, H_out, W_out), dtype=int)

        # iterate over output frame
        for i in range(H_out):
            for j in range(W_out):

                # get window with stride and pool size
                window = x[
                    :,
                    :,
                    i * self.stride : i * self.stride + self.pool_size,
                    j * self.stride : j * self.stride + self.pool_size,
                ]
                # flatten window
                window_flat = window.reshape(N, C, -1)

                # find max value in window and set output accordingly
                out[:, :, i, j] = window_flat.max(axis=2)
                # save max index to argmax cache
                self.argmax_cache[:, :, i, j] = window_flat.argmax(axis=2)

        return out

    def backward(self, dout) -> np.ndarray:
        """
        dout: (N, C, H_out, W_out)
        returns: (N, C, H, W)
        """

        # decode flat index
        N, C, H_out, W_out = dout.shape
        dx = np.zeros(self.x_shape)

        for i in range(H_out):
            for j in range(W_out):
                flat_idx = self.argmax_cache[:, :, i, j]

                row_offset = flat_idx // self.pool_size
                col_offset = flat_idx % self.pool_size

                n_idx, c_idx = np.indices((N, C))
                dx[
                    n_idx,
                    c_idx,
                    i * self.stride + row_offset,
                    j * self.stride + col_offset,
                ] += dout[:, :, i, j]

        return dx


def im2col(x, kernel_h, kernel_w, stride):
    """
    x: (N, C, H, W)
    col: (C * kernel_h * kernel_w, N * out_h * out_w)
        - each column is one flattened receptive field patch
        - columns are ordered by (n, out_row, out_col)
    """

    N, C, H, W = x.shape
    out_h = (H - kernel_h) // stride + 1
    out_w = (W - kernel_w) // stride + 1

    # windows: (N, C, H - kernel_h + 1, W - kernel_w + 1, kernel_h, kernel_w)
    windows = np.lib.stride_tricks.sliding_window_view(
        x, (kernel_h, kernel_w), axis=(2, 3)
    )
    # apply stride by subsampling window-position axes -> (N, C, out_h, out_w, kernel_h, kernel_w)
    windows = windows[:, :, ::stride, ::stride, :, :]

    # reorder to (C, kernel_h, kernel_w, N, out_h, out_w) then flatten each half
    col = windows.transpose(1, 4, 5, 0, 2, 3).reshape(
        C * kernel_h * kernel_w, N * out_h * out_w
    )

    return col


def col2im(col, x_shape, kernel_h, kernel_w, stride):
    """
    col: (C * kernel_h * kernel_w, N * out_h * out_w)
    x_shape: (N, C, H, W)
    returns: (N, C, H, W)
    """

    N, C, H, W = x_shape
    out_h = (H - kernel_h) // stride + 1
    out_w = (W - kernel_w) // stride + 1

    # reshape col to (C, kernel_h, kernel_w, N, out_h, out_w) -- inverse of im2col's transpose+reshape
    col_reshaped = col.reshape(C, kernel_h, kernel_w, N, out_h, out_w)

    # target (row, col) in x for every (kernel offset, output position) pair
    row_idx = np.arange(kernel_h)[:, None] + stride * np.arange(out_h)[None, :]  # (kh, out_h)
    col_idx = np.arange(kernel_w)[:, None] + stride * np.arange(out_w)[None, :]  # (kw, out_w)

    # broadcast every index axis to col_reshaped's shape (C, kh, kw, N, out_h, out_w)
    n_idx = np.arange(N).reshape(1, 1, 1, N, 1, 1)
    c_idx = np.arange(C).reshape(C, 1, 1, 1, 1, 1)
    r_idx = row_idx.reshape(1, kernel_h, 1, 1, out_h, 1)
    w_idx = col_idx.reshape(1, 1, kernel_w, 1, 1, out_w)
    n_b, c_b, r_b, w_b = np.broadcast_arrays(n_idx, c_idx, r_idx, w_idx)

    # scatter-add, not assign: overlapping windows (stride < kernel_size) must accumulate
    x = np.zeros(x_shape, dtype=col.dtype)
    np.add.at(x, (n_b, c_b, r_b, w_b), col_reshaped)

    return x
