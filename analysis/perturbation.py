"""
How perturbations to the input propagate through the network.

Everything here is a direct measurement -- no Gaussian or independence
assumptions -- so it stays meaningful on trained weights, where the mean-field
signal-propagation theory does not apply.
"""

import numpy as np

N_DIRECTIONS = 8
REL_EPS = 1e-3


def per_sample_norm(a):
    return np.linalg.norm(a.reshape(a.shape[0], -1), axis=1)


def normalize(a, eps=1e-30):
    n = np.maximum(per_sample_norm(a), eps)
    return a / n.reshape(-1, *([1] * (a.ndim - 1)))


def scale_to(x, direction, rel_eps=REL_EPS):
    """Scale a unit direction relative to each input's own norm, so the
    measurement doesn't depend on absolute pixel scale."""
    scale = rel_eps * per_sample_norm(x)
    return direction * scale.reshape(-1, *([1] * (x.ndim - 1)))


def difference_norms(model, x, delta):
    """
    Propagate x and x+delta layer by layer, recording the per-sample norm of
    their difference at every layer boundary.

    Returns (n_layers + 1, N) -- row 0 is the input perturbation itself.

    Runs each layer's forward twice, which clobbers the cache it keeps for
    backward. Fine here (nothing calls backward), but don't interleave this
    with training, or with anything that needs those caches.
    """
    a, b = x, x + delta
    n = x.shape[0]
    norms = [np.linalg.norm((b - a).reshape(n, -1), axis=1)]

    for layer in model.layers:
        a = layer.forward(a)
        b = layer.forward(b)
        norms.append(np.linalg.norm((b - a).reshape(n, -1), axis=1))

    return np.array(norms)


def _ratios(norms):
    per_layer = (norms[1:] / np.maximum(norms[:-1], 1e-30)).mean(axis=1)
    cumulative = (norms[-1] / np.maximum(norms[0], 1e-30)).mean()
    return per_layer, cumulative


def expansion_profile(model, x, n_directions=N_DIRECTIONS, rel_eps=REL_EPS, seed=0):
    """
    Expansion factor per layer, ||d^l|| / ||d^(l-1)||, averaged over random
    perturbation directions and over inputs.

    Note this measures GENERIC sensitivity. In 784 dimensions a random direction
    has cosine ~1/sqrt(784) with any specific one, so it barely probes the
    subspace that decides classifications -- see directional_expansion.

    Returns (per_layer_ratios (n_layers,), cumulative_input_to_output (scalar)).
    """
    rng = np.random.default_rng(seed)
    collected = [
        difference_norms(model, x, scale_to(x, normalize(rng.standard_normal(x.shape)), rel_eps))
        for _ in range(n_directions)
    ]
    return _ratios(np.concatenate(collected, axis=1))


def directional_expansion(model, x, direction, rel_eps=REL_EPS):
    """Expansion along one specific unit direction per sample."""
    return _ratios(difference_norms(model, x, scale_to(x, direction, rel_eps)))
