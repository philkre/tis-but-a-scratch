"""
Does the network drift away from its critical initialization during training?

He init sets sigma_w^2 = 2/fan_in, which for ReLU is exactly the mean-field
critical point -- perturbations neither grow nor shrink. Without weight decay,
gradient descent keeps inflating weights once the training set is fit, so the
prediction was a drift toward the chaotic side.

Second hypothesis, since refuted: a trained classifier should be contractive
inside a class and expansive across boundaries, making expansion input-dependent
rather than global.

The mean-field theory assumes random Gaussian weights and does not survive
training, so nothing here computes chi analytically. Everything is measured by
propagating perturbations through the actual network.

    python -m experiments.criticality           # train with snapshots, then analyze
    python -m experiments.criticality analyze   # analyze existing snapshots
"""

import sys

import numpy as np

from nn import SGD, SoftmaxCrossEntropy, build_model, load_mnist, save_checkpoint, train
from analysis.metrics import layer_labels, logit_margins
from analysis.perturbation import expansion_profile
from analysis.weights import weight_norms
from experiments.snapshots import (
    BATCH_SIZE, EPOCHS, LR, N_PROBE,
    load_snapshot, require_snapshots, snapshot_callback, snapshot_path,
)
from viz.sensitivity import plot_expansion_vs_epoch, plot_layer_profiles, plot_weight_norms


def train_with_snapshots():
    x_train, y_train, x_val, y_val, _, _ = load_mnist()
    model = build_model()

    save_checkpoint(model, snapshot_path(0))     # epoch 0 == initialization
    print(f"training with per-epoch snapshots: lr={LR} epochs={EPOCHS}")
    train(
        model, SoftmaxCrossEntropy(), SGD(lr=LR),
        x_train, y_train, x_val, y_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        epoch_callback=snapshot_callback(),
    )


def analyze():
    require_snapshots()
    *_, x_test, y_test = load_mnist()
    model = build_model()
    labels = layer_labels(model)

    # Margin groups come from the FINAL model and stay fixed across epochs.
    # Measuring margin at init would be meaningless -- an untrained network's
    # logits are noise and know nothing about classes.
    load_snapshot(EPOCHS, model)
    margins = logit_margins(model, x_test)
    order = np.argsort(margins)
    probe_low = x_test[order[:N_PROBE]]          # near a decision boundary
    probe_high = x_test[order[-N_PROBE:]]        # deep inside a class

    epochs = list(range(EPOCHS + 1))
    wn_history = {k: [] for k in weight_norms(model)}
    cum_low, cum_high, prof_low, prof_high = [], [], [], []

    for epoch in epochs:
        load_snapshot(epoch, model)
        for k, v in weight_norms(model).items():
            wn_history[k].append(v)

        r_low, c_low = expansion_profile(model, probe_low)
        r_high, c_high = expansion_profile(model, probe_high)
        cum_low.append(c_low)
        cum_high.append(c_high)
        prof_low.append(r_low)
        prof_high.append(r_high)

        print(f"epoch {epoch:2d}  cumulative expansion: low-margin {c_low:.4f}  "
              f"high-margin {c_high:.4f}  ratio {c_low / max(c_high, 1e-30):.3f}")

    plot_weight_norms(epochs, wn_history)
    plot_expansion_vs_epoch(epochs, cum_low, cum_high)
    plot_layer_profiles(labels, prof_low, prof_high)
    print("wrote img/criticality_*.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze()
    else:
        train_with_snapshots()
        analyze()
