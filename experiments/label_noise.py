"""
Label-noise sweep: does any weight-only measure predict generalization?

Every model sees the same 5000 images, has the same architecture and parameter
count, trains for the same number of epochs, and reaches near-100% training
accuracy. Only the fraction of corrupted labels differs. A weight-only measure
that separates these models is measuring something beyond capacity or scale.

    python -m experiments.label_noise pilot    # find the epoch budget
    python -m experiments.label_noise          # full sweep
"""

import os
import sys

import numpy as np

from nn import SGD, SoftmaxCrossEntropy, build_model, evaluate, load_mnist, train
from analysis.weights import collect_measures, snapshot_params

N_TRAIN = 5000
# Accuracy is tracked every epoch, so the full 10k test set would roughly double
# per-epoch cost. 2000 images give about +/-1% precision, which is far finer
# than rank correlation over 25 models needs.
N_TEST = 2000
NOISE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 120
BATCH_SIZE = 32
# Memorizing random labels needs a narrower learning rate than clean training.
# A probe at 100% noise: lr=0.1 oscillated between 0.71 and 0.96 without
# converging, lr=0.02 and 0.01 were too slow to memorize at all, lr=0.05
# reached 1.000 by epoch 80 and settled. The budget gives 50% margin on that.
LR = 0.05
RESULTS_PATH = "results/label_noise.npz"


def corrupt_labels(y_onehot, fraction, rng):
    """
    Replace `fraction` of labels with uniformly random ones.

    Uniform over all 10 classes, following Zhang et al. -- so at fraction=1.0
    roughly 10% of labels land on the correct class by chance. Returns a new
    array; the input is not modified.
    """
    y = y_onehot.copy()
    n = y.shape[0]
    n_corrupt = int(round(fraction * n))
    if n_corrupt == 0:
        return y
    idx = rng.choice(n, n_corrupt, replace=False)
    y[idx] = np.eye(10)[rng.integers(0, 10, n_corrupt)]
    return y


def run_one(x_train, y_train, x_test, y_test, noise, seed, epochs, verbose=False):
    """
    Train one model and record every weight measure at every epoch.

    Returns a list of per-epoch dicts. Test accuracy is recorded as ground
    truth for later validation -- collect_measures never receives it.
    """
    rng = np.random.default_rng(1000 + seed)
    y_noisy = corrupt_labels(y_train, noise, rng)

    model = build_model(seed=seed)
    init_params = snapshot_params(model)
    loss_fn = SoftmaxCrossEntropy()
    optimizer = SGD(lr=LR)
    rows = []

    def on_epoch_end(epoch, m):
        train_loss, train_acc = evaluate(m, loss_fn, x_train, y_noisy)
        _, test_acc = evaluate(m, loss_fn, x_test, y_test)
        row = {"noise": noise, "seed": seed, "epoch": epoch,
               "train_loss": train_loss, "train_acc": train_acc, "test_acc": test_acc}
        row.update(collect_measures(m, init_params))
        rows.append(row)
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"  epoch {epoch:3d}  train_acc {train_acc:.4f}  test_acc {test_acc:.4f}",
                  flush=True)

    train(
        model, loss_fn, optimizer,
        x_train, y_noisy, x_test, y_test,
        epochs=epochs, batch_size=BATCH_SIZE,
        checkpoint_path=None, seed=seed, epoch_callback=on_epoch_end,
    )
    return rows


def _data():
    x_train, y_train, _, _, x_test, y_test = load_mnist()
    return (x_train[:N_TRAIN], y_train[:N_TRAIN],
            x_test[:N_TEST], y_test[:N_TEST])


def pilot():
    """
    Fix the epoch budget. 100% noise is the slowest case to memorize, so
    whatever it needs sets the budget for every condition -- matched training
    accuracy is what makes the comparison clean.
    """
    x_train, y_train, x_test, y_test = _data()
    print(f"pilot: 100% label noise, {N_TRAIN} images, up to {EPOCHS} epochs", flush=True)
    rows = run_one(x_train, y_train, x_test, y_test,
                   noise=1.0, seed=0, epochs=EPOCHS, verbose=True)

    reached = [r["epoch"] for r in rows if r["train_acc"] >= 0.99]
    final = rows[-1]
    if reached:
        print(f"\nreached 99% train accuracy at epoch {reached[0]} "
              f"(final train {final['train_acc']:.4f}, test {final['test_acc']:.4f})")
        print(f"set EPOCHS to at least {reached[0]} in this file before the full sweep")
    else:
        print(f"\nDID NOT memorize within {EPOCHS} epochs "
              f"(final train accuracy {final['train_acc']:.4f})")
        print("raise EPOCHS, or lower N_TRAIN -- do not shorten the budget instead")


def sweep():
    """All noise levels x all seeds. Writes one record for later analysis."""
    x_train, y_train, x_test, y_test = _data()

    all_rows = []
    total = len(NOISE_LEVELS) * len(SEEDS)
    for i, noise in enumerate(NOISE_LEVELS):
        for j, seed in enumerate(SEEDS):
            n = i * len(SEEDS) + j + 1
            print(f"[{n}/{total}] noise={noise:.2f} seed={seed}", flush=True)
            rows = run_one(x_train, y_train, x_test, y_test, noise, seed, EPOCHS)
            print(f"    final train_acc {rows[-1]['train_acc']:.4f}  "
                  f"test_acc {rows[-1]['test_acc']:.4f}", flush=True)
            all_rows.extend(rows)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    record = {k: np.array([r[k] for r in all_rows]) for k in all_rows[0]}
    np.savez(RESULTS_PATH, **record)
    print(f"\nwrote {RESULTS_PATH}: {len(all_rows)} rows, {len(record)} columns")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pilot":
        pilot()
    else:
        sweep()
