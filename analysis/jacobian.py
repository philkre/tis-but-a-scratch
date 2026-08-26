"""
The input-output Jacobian and its spectrum.

The network maps R^784 -> R^10, so J is only 10x784 and can be extracted
exactly with ten backward passes -- no approximation, no sampling.
"""

import numpy as np

from nn import N_CLASSES
from analysis.metrics import top_two_classes
from analysis.perturbation import normalize, per_sample_norm, scale_to, REL_EPS

POWER_ITERS = 8


def jacobian(model, x):
    """
    Exact (N, 10, D) input-output Jacobian.

    backward() only reads the forward caches, never writes them, so a single
    forward supports all ten rows.
    """
    n = x.shape[0]
    model.forward(x)
    rows = []
    for i in range(N_CLASSES):
        dout = np.zeros((n, N_CLASSES))
        dout[:, i] = 1.0
        rows.append(model.backward(dout).reshape(n, -1))
    return np.stack(rows, axis=1)


def spectrum(model, x):
    """
    Per-sample singular values of the Jacobian, shape (N, 10), descending.

    These are the network's expansion factors: sigma_1 is how much it amplifies
    its most sensitive input direction, sigma_10 its least. Every remaining
    input direction sits at exactly zero, since the output is 10-dimensional.
    """
    return np.linalg.svd(jacobian(model, x), compute_uv=False)


def effective_rank(sigmas):
    """
    exp(entropy) of the normalized singular-value energy: how many directions
    the network meaningfully uses, out of the 10 available.
    """
    energy = sigmas**2
    p = energy / energy.sum(axis=1, keepdims=True)
    return np.exp(-(p * np.log(np.maximum(p, 1e-30))).sum(axis=1))


def random_direction_expansion(sigmas, d_in):
    """
    E[||Jv||] over unit random v, computed exactly from the spectrum as
    ||J||_F / sqrt(d_in) -- no sampling needed.

    Averaging this way is what makes a wide spectrum look like a single
    expansion factor; compare against sigmas[:, 0].
    """
    return np.sqrt((sigmas**2).sum(axis=1) / d_in)


def margin_direction(model, x, pairs):
    """
    Unit-norm gradient of (logit_a - logit_b) w.r.t. the input, per sample --
    the steepest boundary-crossing direction for that class pair.

    Sequential.backward propagates dL/d(output) back to dL/d(input), so seeding
    it with (e_a - e_b) yields exactly that gradient; no loss layer involved.

    Leaves the layer caches populated for x. Anything that runs forward again
    invalidates them, so call this before such work, not after.
    """
    n = x.shape[0]
    model.forward(x)
    dout = np.zeros((n, N_CLASSES))
    rows = np.arange(n)
    dout[rows, pairs[:, 0]] = 1.0
    dout[rows, pairs[:, 1]] = -1.0
    return normalize(model.backward(dout))


def margin_gradient_norms(model, x):
    """Raw ||grad m|| per sample, where m is the top-1 minus top-2 logit gap."""
    pairs = top_two_classes(model, x)
    n = x.shape[0]
    model.forward(x)
    dout = np.zeros((n, N_CLASSES))
    rows = np.arange(n)
    dout[rows, pairs[:, 0]] = 1.0
    dout[rows, pairs[:, 1]] = -1.0
    return per_sample_norm(model.backward(dout))


def top_singular_value(model, x, n_iter=POWER_ITERS, rel_eps=REL_EPS, seed=0):
    """
    Largest singular value per sample by power iteration on J^T J.

    J @ v comes from a finite difference -- the network is piecewise linear, so
    for a step small enough to stay inside one linear region that difference is
    exact, not an approximation. J^T @ u comes from backward().

    Layers process samples independently, so J is block-diagonal across the
    batch and per-sample normalization keeps each sample's iteration separate.

    Cheaper than a full SVD when only the top value is needed; spectrum() is
    the better choice when the whole spectrum matters.

    Returns (sigma_per_sample, top_direction).
    """
    rng = np.random.default_rng(seed)
    v = normalize(rng.standard_normal(x.shape))
    sigma_sq = np.ones(x.shape[0])

    for _ in range(n_iter):
        delta = scale_to(x, v, rel_eps)
        step = per_sample_norm(delta)

        f_plus = model.forward(x + delta)
        f_zero = model.forward(x)          # also restores the caches for x
        u = (f_plus - f_zero) / step[:, None]

        JtJv = model.backward(u)            # J^T @ (J @ v)
        sigma_sq = np.maximum(per_sample_norm(JtJv), 1e-30)
        v = JtJv / sigma_sq.reshape(-1, *([1] * (x.ndim - 1)))

    return np.sqrt(sigma_sq), v
