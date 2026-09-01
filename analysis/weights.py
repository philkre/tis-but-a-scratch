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


def power_law_alpha(model, tail_frac=0.5):
    """
    Hill-estimator tail exponent of each layer's eigenvalue spectrum.

    Eigenvalues of W^T W are the squared singular values. Taking the largest
    `tail_frac` of them, the maximum-likelihood exponent of a power law
    p(x) ~ x^-alpha above x_min is

        alpha = 1 + n / sum(ln(x_i / x_min))

    Statistically fragile at this scale -- the largest weight matrix here has
    120 eigenvalues, where published work uses matrices orders of magnitude
    larger. Report it; don't let it lead a conclusion.
    """
    out = {}
    for name, s in singular_values(model).items():
        eig = np.sort(s**2)[::-1]
        k = max(int(len(eig) * tail_frac), 2)
        tail = eig[:k]
        x_min = tail[-1]
        logs = np.log(np.maximum(tail / max(x_min, 1e-30), 1.0 + 1e-12))
        out[name] = float(1.0 + len(tail) / max(logs.sum(), 1e-30))
    return out


MEASURES = ("weight_norm", "distance_from_init", "stable_rank",
            "spectral_entropy", "power_law_alpha")


def collect_measures(model, init_params):
    """
    Every weight-only measure, flattened to {"measure/layer": value}.

    Takes a model and its initial weights -- and nothing else. There is
    deliberately no parameter through which data could enter, so no measure
    can accidentally consume the test set it is meant to predict.
    """
    sources = {
        "weight_norm": weight_norms(model),
        "distance_from_init": distance_from_init(model, init_params),
        "stable_rank": stable_rank(model),
        "spectral_entropy": spectral_entropy(model),
        "power_law_alpha": power_law_alpha(model),
    }
    return {f"{measure}/{layer}": value
            for measure, per_layer in sources.items()
            for layer, value in per_layer.items()}


def _check_power_law_alpha():
    """
    Hill estimator against synthetic data with a known exponent.

    Inverse-transform sampling: if x = x_min * (1-u)^(-1/(alpha-1)) for uniform
    u, then x follows a power law with that exponent.
    """
    rng = np.random.default_rng(0)
    true_alpha = 3.0
    u = rng.random(20000)
    x = (1 - u) ** (-1 / (true_alpha - 1))

    tail = np.sort(x)[::-1][:10000]
    x_min = tail[-1]
    logs = np.log(np.maximum(tail / x_min, 1.0 + 1e-12))
    est = 1.0 + len(tail) / logs.sum()

    assert abs(est - true_alpha) < 0.15, f"Hill estimator gave {est:.3f}, expected ~{true_alpha}"
    print(f"power_law_alpha check passed (recovered alpha={est:.3f} from synthetic alpha={true_alpha})")
    return True
