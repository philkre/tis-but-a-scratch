"""
Figures for the Jacobian-spectrum result.

The claim these support: a trained network has no single order parameter. Its
input-output Jacobian spans a wide range of expansion factors, so the scalar
chi of mean-field theory averages away the structure that matters.
"""

import numpy as np
import matplotlib.pyplot as plt

from viz.style import AQUA, BLUE, GRID, INK_2, ORANGE, RAMP, legend, save, style

N_CLASS = 10
D_IN = 784


def plot_spectrum(spectra, ramp_epochs, save_path="img/spectrum.png"):
    """Headline: no single expansion factor describes the trained network."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    idx = np.arange(1, N_CLASS + 1)

    lo = min(spectra[e]["sigma"].min() for e in ramp_epochs) * 0.6
    hi = max(spectra[e]["sigma"].max() for e in ramp_epochs) * 1.5
    ax.axhspan(lo, 1.0, color=GRID, alpha=0.35, zorder=0, linewidth=0)
    ax.axhline(1.0, color=INK_2, linestyle="--", linewidth=1.1, zorder=1)

    for epoch, color in zip(ramp_epochs, RAMP):
        ax.plot(idx, spectra[epoch]["sigma"], color=color, linewidth=1.8,
                marker="o", markersize=4.5, label=f"epoch {epoch}", zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    ax.set_xticks(idx)
    style(ax,
          title="No single order parameter: the trained network amplifies its most\n"
                "sensitive direction 13x more than its least",
          xlabel="singular value index",
          ylabel=r"$\sigma$   (output change per unit input change)")

    ax.text(10.25, 1.0, "  $\\sigma=1$\n  no amplification", color=INK_2,
            fontsize=8.5, va="center", ha="left")
    ax.text(0.98, 0.03,
            f"{D_IN - N_CLASS} further input directions sit at exactly $\\sigma=0$ —\n"
            "the output is only 10-dimensional",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=INK_2)
    legend(ax, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    save(fig, save_path)


def plot_spectrum_over_training(spectra, save_path="img/spectrum_over_training.png"):
    """Left: the average is an artifact of a spreading spectrum. Right: effective rank."""
    epochs = sorted(spectra)
    s1 = np.array([spectra[e]["sigma"][0] for e in epochs])
    s10 = np.array([spectra[e]["sigma"][-1] for e in epochs])
    rand = np.array([spectra[e]["random"] for e in epochs])
    eff = np.array([spectra[e]["eff_rank"] for e in epochs])

    fig, (ax, ax_r) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax.fill_between(epochs, s10, s1, color=BLUE, alpha=0.12, linewidth=0)
    ax.plot(epochs, s1, color=BLUE, linewidth=1.8, marker="o", markersize=4.5,
            label=r"$\sigma_1$  (most expansive direction)")
    ax.plot(epochs, s10, color=AQUA, linewidth=1.8, marker="s", markersize=4.5,
            label=r"$\sigma_{10}$  (least expansive)")
    ax.plot(epochs, rand, color=ORANGE, linewidth=1.8, marker="^", markersize=4.5,
            label="mean over random directions")
    ax.axhline(1.0, color=INK_2, linestyle="--", linewidth=1.1)
    ax.set_yscale("log")
    ax.set_xticks(epochs[::2])
    style(ax, title="Averaging over directions collapses a 13x spread to one number",
          xlabel="epoch", ylabel="expansion factor")
    legend(ax, loc="lower right")

    ax_r.axhline(N_CLASS, color=GRID, linewidth=1.2)
    ax_r.text(epochs[-1], N_CLASS, " 10 available", color=INK_2, fontsize=8.5,
              va="center", ha="left")
    ax_r.plot(epochs, eff, color=BLUE, linewidth=1.8, marker="o", markersize=4.5)
    ax_r.set_ylim(0, 11.5)
    ax_r.set_xticks(epochs[::2])
    style(ax_r, title="Effective rank collapses within one epoch",
          xlabel="epoch", ylabel="effective rank of the Jacobian")

    save(fig, save_path)


def plot_margin_vs_gradient(margins, gnorms, save_path="img/margin_vs_gradient.png"):
    """The refuted hypothesis: local sensitivity is flat across confidence."""
    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.scatter(margins, gnorms, s=5, color=BLUE, alpha=0.18, linewidths=0,
               label="test image")

    bins = np.logspace(np.log10(max(margins.min(), 1e-3)), np.log10(margins.max()), 14)
    centers, med = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (margins >= a) & (margins < b)
        if m.sum() >= 8:
            centers.append(np.sqrt(a * b))
            med.append(np.median(gnorms[m]))
    ax.plot(centers, med, color=ORANGE, linewidth=2.2, marker="o", markersize=5,
            label="median per bin")

    r = np.corrcoef(np.log(np.maximum(margins, 1e-3)), np.log(gnorms))[0, 1]
    ax.set_xscale("log")
    ax.set_ylim(0, np.percentile(gnorms, 99.5) * 1.1)
    style(ax,
          title="Confidence comes from distance, not from flatness:\n"
                f"local sensitivity is flat across the full range of margins  (log-log r = {r:+.2f})",
          xlabel="margin  (top-1 minus runner-up logit)",
          ylabel=r"$\|\nabla m\|$   local sensitivity")
    legend(ax, fontsize=9, loc="upper left")
    save(fig, save_path)
