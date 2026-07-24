import numpy as np
from layers import *


def numerical_gradient(f, x, eps=1e-5) -> np.ndarray:
    """
    f: callable, takes x (ndarray) and any fixed extra args, returns scalar
    x: ndarray, shape (N, ...), the point to compute the gradient at

    Returns: ndarray, shape (N, ...), the finite-difference gradient of f at x
    """

    # gradient
    grad = np.zeros_like(x)

    # iterate over all indices in x
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = x[idx]

        x[idx] = orig + eps
        f_plus = f(x)

        x[idx] = orig - eps
        f_minus = f(x)

        x[idx] = orig

        grad[idx] = (f_plus - f_minus) / (2 * eps)

    return grad


def relative_error(a, b):
    return np.max(np.abs(a - b) / (np.maximum(1e-8, np.abs(a) + np.abs(b))))


def check_layer(layer, x, tol=1e-5, seed=1004) -> bool:
    """
    Given a layer instance and a random small input x:
    - run layer.forward(x) to get output, then layer.backward(dout) for a random dout of matching shape (this is the analytical gradient)
    - compute numerical gradiant of scalar loss L = sum(out*dout) w.r.t. x using numerical_gradient()
    - assert relativer error < tol for dx and for every param gradient
    - raise AssertionError if any check fails, otherwise return True
    """

    # compute analytical gradients
    rng = np.random.default_rng(seed)
    fwd = layer.forward(x)
    dout = rng.standard_normal(fwd.shape)
    bwd = layer.backward(dout)
    analytical_dx = bwd.copy()
    analytical_grads = {k: v.copy() for k, v in layer.grads.items()}

    # compute numerical gradient
    def loss_wrt_x(x):
        return np.sum(layer.forward(x) * dout)

    numerical_dx = numerical_gradient(loss_wrt_x, x)
    numerical_grads = {}
    for name, w in layer.params.items():

        def param_loss_fn(_):
            return np.sum(layer.forward(x) * dout)

        numerical_grads[name] = numerical_gradient(param_loss_fn, w)

    # check relative error
    dx_error = relative_error(analytical_dx, numerical_dx)
    assert dx_error < tol, f"dx relative error {dx_error} exceeds tolerance {tol}"

    # check relative error for each parameter gradient
    for name, analytical_grad in analytical_grads.items():
        numerical_grad = numerical_grads[name]
        grad_error = relative_error(analytical_grad, numerical_grad)
        assert (
            grad_error < tol
        ), f"gradient {name} relative error {grad_error} exceeds tolerance {tol}"

    print("All gradient checks passed.")
    return True


def check_dense():
    """
    Check Dense layer gradients using check_layer.
    """
    in_feats = 5
    out_feats = 3
    layer = Dense(in_feats, out_feats)
    x = np.random.randn(4, in_feats)
    return check_layer(layer, x)


def check_relu():
    """
    Check ReLU layer gradients using check_layer.
    """
    layer = ReLU()
    x = np.random.randn(4, 5) + 0.5
    return check_layer(layer, x)


def check_softmax_ce(x, y, tol=1e-5):
    """
    Check SoftmaxCrossEntropy layer gradients using numerical gradient checking.
    """
    layer = SoftmaxCrossEntropy()
    layer.forward(x, y)
    analytical_dx = layer.backward().copy()

    def loss_wrt_x(x_):
        return layer.forward(x_, y)

    numerical_dx = numerical_gradient(loss_wrt_x, x)

    error = relative_error(analytical_dx, numerical_dx)
    assert error < tol, f"dx relative error {error} exceeds tolerance {tol}"
    print("SoftmaxCrossEntropy gradient check passed.")
    return True


def check_softmax():
    """
    Check softmax layer
    """
    x = np.random.randn(4, 10)
    y = np.zeros_like(x)
    y[np.arange(4), np.random.randint(0, 10, size=4)] = 1
    return check_softmax_ce(x, y)


def check_flatten():
    """
    Check Flatten layer gradients using check_layer.
    """
    layer = Flatten()
    x = np.random.randn(4, 3, 2, 3)
    return check_layer(layer, x)


def check_maxpool():
    """
    Check Maxpool2D gradients
    """
    x = np.random.randn(2, 3, 4, 4)
    x += np.arange(x.size).reshape(x.shape) * 1e-3  # ensure unique values for maxpool
    layer = MaxPool2D()
    return check_layer(layer, x)


def check_im2col2im():
    """
    Check im2col and col2im functions for consistency.
    """
    x = np.random.randn(2, 3, 4, 4)
    kernel_h = 2
    kernel_w = 2
    stride = 2

    # Convert to columns
    cols = im2col(x, kernel_h, kernel_w, stride)

    # Convert back to image
    x_reconstructed = col2im(cols, x.shape, kernel_h, kernel_w, stride)

    # Check if the reconstructed image is close to the original
    error = relative_error(x, x_reconstructed)
    assert error < 1e-5, f"im2col/col2im reconstruction error {error} exceeds tolerance"
    print("im2col and col2im check passed.")
    return True


def check_conv2d():
    """
    Check Conv2D layer gradients using check_layer.
    """
    in_channels = 3
    out_channels = 2
    kernel_size = 3
    stride = 1
    layer = Conv2D(in_channels, out_channels, kernel_size, stride)
    x = np.random.randn(2, in_channels, 5, 5)
    return check_layer(layer, x)


def check_all():
    """
    Run all gradient checks.
    """
    check_dense()
    check_relu()
    check_softmax()
    check_flatten()
    check_maxpool()
    check_im2col2im()
    check_conv2d()
