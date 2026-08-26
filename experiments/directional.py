"""
Sensitivity along directions that matter, rather than random ones.

experiments/criticality.py found no difference between near-boundary and
deep-in-class inputs. Likely because it could not: a random direction in 784
dimensions has cosine ~1/sqrt(784) with any specific one, so random probes
barely touch the subspace that crosses a decision boundary.

Three probes here:
    random  -- baseline, same as before
    margin  -- gradient of (logit_a - logit_b), the steepest boundary crossing
    top     -- largest Jacobian singular value; every other measurement must
               come in below it, which doubles as a correctness check

Class pairs are fixed from the FINAL model and held constant across epochs, so
every snapshot answers the same question rather than a moving one.

    python -m experiments.directional
"""

import numpy as np

from nn import build_model, load_mnist
from analysis.jacobian import margin_direction, top_singular_value
from analysis.metrics import logit_margins, top_two_classes
from analysis.perturbation import directional_expansion, expansion_profile
from experiments.snapshots import EPOCHS, N_PROBE, load_snapshot, require_snapshots
from viz.sensitivity import plot_directional


def analyze():
    require_snapshots()
    *_, x_test, y_test = load_mnist()
    model = build_model()

    load_snapshot(EPOCHS, model)
    margins = logit_margins(model, x_test)
    order = np.argsort(margins)
    groups = {
        "low margin": x_test[order[:N_PROBE]],
        "high margin": x_test[order[-N_PROBE:]],
    }
    pairs = {name: top_two_classes(model, xs) for name, xs in groups.items()}

    epochs = list(range(EPOCHS + 1))
    results = {name: {"random": [], "margin": [], "top": []} for name in groups}

    for epoch in epochs:
        load_snapshot(epoch, model)
        line = [f"epoch {epoch:2d}"]

        for name, xs in groups.items():
            # margin direction first -- later forwards clobber the caches
            d_margin = margin_direction(model, xs, pairs[name])
            _, c_margin = directional_expansion(model, xs, d_margin)
            _, c_random = expansion_profile(model, xs)
            c_top = float(top_singular_value(model, xs)[0].mean())

            results[name]["random"].append(float(c_random))
            results[name]["margin"].append(float(c_margin))
            results[name]["top"].append(c_top)
            line.append(f"| {name:11s} rand {c_random:.3f} margin {c_margin:.3f} top {c_top:.3f}")

        print("  ".join(line))

    for name in groups:
        r = results[name]
        for key in ("random", "margin"):
            over = [e for e, a, b in zip(epochs, r[key], r["top"]) if a > b * 1.02]
            if over:
                print(f"WARNING: {name}/{key} exceeds the top singular value at epochs {over}")

    plot_directional(epochs, results)
    print("wrote img/directional_*.png")


if __name__ == "__main__":
    analyze()
