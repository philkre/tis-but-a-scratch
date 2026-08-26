"""
The Jacobian spectrum: why a scalar order parameter does not describe this
network, and why the apparent drift toward criticality was an averaging artifact.

    python -m experiments.spectrum
"""

import numpy as np

from nn import INPUT_SHAPE, build_model, load_mnist
from analysis.jacobian import effective_rank, margin_gradient_norms, random_direction_expansion, spectrum
from analysis.metrics import logit_margins
from experiments.snapshots import EPOCHS, load_snapshot, require_snapshots
from viz.spectrum import plot_margin_vs_gradient, plot_spectrum, plot_spectrum_over_training

RAMP_EPOCHS = [0, 1, 2, 6, 12]      # five steps: matches the validated ordinal ramp
N_PROBE = 128
N_SCATTER = 3000
D_IN = int(np.prod(INPUT_SHAPE))


def collect_spectra(xs, epochs):
    model = build_model()
    out = {}
    for epoch in epochs:
        load_snapshot(epoch, model)
        s = spectrum(model, xs)
        out[epoch] = {
            "sigma": s.mean(axis=0),
            "random": random_direction_expansion(s, D_IN).mean(),
            "eff_rank": effective_rank(s).mean(),
        }
    return out


def analyze():
    require_snapshots()
    *_, x_test, y_test = load_mnist()

    spectra = collect_spectra(x_test[:N_PROBE], list(range(EPOCHS + 1)))
    plot_spectrum(spectra, RAMP_EPOCHS)
    plot_spectrum_over_training(spectra)

    model = load_snapshot(EPOCHS)
    xs = x_test[:N_SCATTER]
    margins = logit_margins(model, xs)
    gnorms = np.concatenate([
        margin_gradient_norms(model, xs[i : i + 256]) for i in range(0, len(xs), 256)
    ])
    plot_margin_vs_gradient(margins, gnorms)

    print(f"effective rank: {spectra[0]['eff_rank']:.2f} (init) -> {spectra[EPOCHS]['eff_rank']:.2f}")
    print(f"sigma_1: {spectra[0]['sigma'][0]:.2f} -> {spectra[EPOCHS]['sigma'][0]:.2f}")
    print(f"sigma_1 / sigma_10 (trained): {spectra[EPOCHS]['sigma'][0] / spectra[EPOCHS]['sigma'][-1]:.1f}x")
    print("wrote img/spectrum.png, img/spectrum_over_training.png, img/margin_vs_gradient.png")


if __name__ == "__main__":
    analyze()
