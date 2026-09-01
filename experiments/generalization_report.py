"""
Does any weight-only measure predict generalization?

Reads results/label_noise.npz and answers three questions:
  1. Does the measure rank models by test accuracy?          (Kendall tau)
  2. Does it beat the weight-norm baseline?                  (partial correlation)
  3. Does it predict early, or only describe the endpoint?   (tau at earlier epochs)

    python -m experiments.generalization_report
"""

import numpy as np

from analysis.generalization import kendall_tau, partial_spearman
from analysis.weights import MEASURES
from viz.generalization import plot_measure_trajectories, plot_measure_vs_accuracy

RESULTS_PATH = "results/label_noise.npz"
BASELINE = "weight_norm"
EARLY_EPOCHS = (10, 25, 50)
TAU_THRESHOLD = 0.5


def _aggregate(record, measure, mask):
    """Mean of a measure across layers, one value per model."""
    cols = [k for k in record.files if k.startswith(f"{measure}/")]
    return np.mean([record[c][mask] for c in cols], axis=0)


def main():
    record = np.load(RESULTS_PATH)
    epoch, noise, test_acc = record["epoch"], record["noise"], record["test_acc"]
    final = epoch == epoch.max()

    y = test_acc[final]
    baseline = _aggregate(record, BASELINE, final)
    aggregated = {m: _aggregate(record, m, final) for m in MEASURES}

    print(f"{len(y)} models.  test accuracy {y.min():.3f} - {y.max():.3f}\n")
    print(f"{'measure':22s} {'tau':>7s} {'partial|norm':>13s}")
    taus = {}
    for name, values in aggregated.items():
        taus[name] = kendall_tau(values, y)
        partial = "     (baseline)" if name == BASELINE else \
            f"{partial_spearman(values, y, baseline):+13.3f}"
        print(f"{name:22s} {taus[name]:+7.3f} {partial}")

    print(f"\nper layer, Kendall tau against test accuracy:")
    layers = sorted({k.split("/", 1)[1] for k in record.files if "/" in k})
    print(f"{'measure':22s} " + " ".join(f"{l:>16s}" for l in layers))
    for name in MEASURES:
        row = [f"{kendall_tau(record[f'{name}/{l}'][final], y):+16.3f}" for l in layers]
        print(f"{name:22s} " + " ".join(row))

    print(f"\ndoes it predict early? Kendall tau vs FINAL test accuracy:")
    print(f"{'measure':22s} " + " ".join(f"{'ep ' + str(e):>10s}" for e in EARLY_EPOCHS))
    for name in MEASURES:
        row = []
        for e in EARLY_EPOCHS:
            mask = epoch == e
            if mask.sum() == final.sum():
                row.append(f"{kendall_tau(_aggregate(record, name, mask), y):+10.3f}")
            else:
                row.append(f"{'n/a':>10s}")
        print(f"{name:22s} " + " ".join(row))

    winners = [n for n, t in taus.items()
               if n != BASELINE and abs(t) > TAU_THRESHOLD
               and abs(partial_spearman(aggregated[n], y, baseline)) > 0.3]
    print("\n" + ("beats the baseline: " + ", ".join(winners) if winners else
                  "NO measure beat the weight-norm baseline. That is the result."))

    plot_measure_vs_accuracy(aggregated, y, noise[final], taus)

    best = max(MEASURES, key=lambda m: abs(taus[m]))
    epochs = sorted(set(epoch))
    per_noise = {
        level: [_aggregate(record, best, (epoch == e) & (noise == level)).mean()
                for e in epochs]
        for level in sorted(set(noise))
    }
    plot_measure_trajectories(epochs, per_noise, best)
    print(f"wrote img/generalization_measures.png, img/generalization_trajectory.png")


if __name__ == "__main__":
    main()
