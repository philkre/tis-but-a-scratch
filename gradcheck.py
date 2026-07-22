import numpy as np


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

    def relative_error(a, b):
        return np.max(np.abs(a - b) / (np.maximum(1e-8, np.abs(a) + np.abs(b))))

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

    return True
