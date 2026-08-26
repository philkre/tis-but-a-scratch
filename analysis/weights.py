"""
Statistics computed from the weights alone -- no data, no forward pass.

These are the candidates for a white-box measure of generalization: anything
here can be evaluated without ever touching a test set.
"""

import numpy as np


def _weighted_layers(model):
    for i, layer in enumerate(model.layers):
        if "W" in layer.params:
            yield f"{i}:{type(layer).__name__}", layer.params["W"]


def weight_norms(model):
    """Frobenius norm of every weight matrix, keyed by position and layer type."""
    return {name: float(np.linalg.norm(W)) for name, W in _weighted_layers(model)}


def as_matrix(W):
    """Flatten a conv kernel (out, in, kh, kw) to (out, in*kh*kw); Dense is already 2D."""
    return W.reshape(W.shape[0], -1) if W.ndim > 2 else W


def singular_values(model):
    """Descending singular values of each layer's weight matrix."""
    return {name: np.linalg.svd(as_matrix(W), compute_uv=False)
            for name, W in _weighted_layers(model)}


def stable_rank(model):
    """
    ||W||_F^2 / ||W||_2^2 per layer -- how many directions the weight matrix
    spreads its energy over. Robust: no fitting, no free parameters.
    """
    out = {}
    for name, s in singular_values(model).items():
        out[name] = float((s**2).sum() / max(s[0] ** 2, 1e-30))
    return out


def spectral_entropy(model):
    """exp(entropy) of each layer's normalized singular-value energy."""
    out = {}
    for name, s in singular_values(model).items():
        p = s**2 / max((s**2).sum(), 1e-30)
        out[name] = float(np.exp(-(p * np.log(np.maximum(p, 1e-30))).sum()))
    return out


def distance_from_init(model, init_params):
    """
    ||W - W_0||_F per layer.

    init_params comes from snapshot_params() on the epoch-0 model. Distance
    travelled in weight space has theoretical backing as a generalization-
    relevant quantity, and is the cheapest candidate to compute.
    """
    return {name: float(np.linalg.norm(W - init_params[name]))
            for name, W in _weighted_layers(model)}


def snapshot_params(model):
    """Copy of every weight matrix, for later comparison against."""
    return {name: W.copy() for name, W in _weighted_layers(model)}
