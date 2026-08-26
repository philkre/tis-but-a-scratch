"""Scalar summaries of a model's behaviour on data. No plotting."""

import numpy as np

from nn import N_CLASSES


def confusion_matrix(model, X, y, batch_size=256):
    """
    Forward pass, argmax predictions vs argmax(y).
    Returns a (10, 10) int ndarray, rows = true label, cols = predicted label.
    """
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for start in range(0, X.shape[0], batch_size):
        xb = X[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model.forward(xb)
        preds = np.argmax(logits, axis=1)
        labels = np.argmax(yb, axis=1)
        np.add.at(cm, (labels, preds), 1)
    return cm


def logit_margins(model, X, batch_size=256):
    """Top-1 minus top-2 logit. Small margin == close to a decision boundary."""
    margins = []
    for start in range(0, X.shape[0], batch_size):
        logits = model.forward(X[start : start + batch_size])
        top2 = np.sort(logits, axis=1)[:, -2:]
        margins.append(top2[:, 1] - top2[:, 0])
    return np.concatenate(margins)


def top_two_classes(model, x):
    """(N, 2) array of each input's top-1 and runner-up class."""
    logits = model.forward(x)
    order = np.argsort(logits, axis=1)
    return np.stack([order[:, -1], order[:, -2]], axis=1)


def layer_labels(model):
    """Stable per-layer labels for plots and result dicts."""
    return [f"{i}:{type(l).__name__}" for i, l in enumerate(model.layers)]
