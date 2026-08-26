"""
Figures for the input-output sensitivity experiments.

Weight-norm growth, expansion factors across training, and per-layer
expansion profiles split by how close an input sits to a decision boundary.
"""

import numpy as np
import matplotlib.pyplot as plt

from viz.style import AQUA, BLUE, GRID, INK_2, ORANGE, legend, save, style


def plot_weight_norms(epochs, wn_history, save_path="img/criticality_weight_norms.png"):
    """One line per weighted layer. Without weight decay these grow without bound."""
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, (label, values) in enumerate(wn_history.items()):
        ax.plot(epochs, values, linewidth=1.8, marker="o", markersize=4,
                label=label, color=plt.get_cmap("Blues")(0.35 + 0.55 * i / max(len(wn_history) - 1, 1)))
    style(ax, title="Weight norms grow monotonically (no weight decay)",
          xlabel="epoch", ylabel=r"$\|W\|_F$")
    legend(ax, fontsize=8)
    save(fig, save_path)


def plot_expansion_vs_epoch(epochs, cum_low, cum_high,
                            save_path="img/criticality_expansion.png"):
    """Sensitivity for near-boundary vs deep-in-class inputs, and their ratio."""
    fig, (ax, ax_ratio) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax.plot(epochs, cum_low, color=BLUE, linewidth=1.8, marker="o", markersize=4,
            label="low margin (near boundary)")
    ax.plot(epochs, cum_high, color=ORANGE, linewidth=1.8, marker="s", markersize=4,
            label="high margin (deep in class)")
    ax.set_yscale("log")
    style(ax, title="Input-output sensitivity",
          xlabel="epoch", ylabel=r"$\|\delta_{out}\| / \|\delta_{in}\|$")
    legend(ax)

    ratio = np.array(cum_low) / np.maximum(np.array(cum_high), 1e-30)
    ax_ratio.axhline(1.0, color=INK_2, linestyle="--", linewidth=1.1)
    ax_ratio.plot(epochs, ratio, color=BLUE, linewidth=1.8, marker="o", markersize=4)
    style(ax_ratio, title="Geometric differentiation  (1.0 = groups behave identically)",
          xlabel="epoch", ylabel="low-margin / high-margin")
    save(fig, save_path)


def plot_layer_profiles(labels, prof_low, prof_high,
                        save_path="img/criticality_layer_profile.png"):
    """Per-layer expansion at init vs after training, both margin groups."""
    x = np.arange(len(labels))
    fig, (ax_init, ax_final) = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)

    for ax, idx, title in (
        (ax_init, 0, "At initialization (epoch 0)"),
        (ax_final, -1, f"After training (epoch {len(prof_low) - 1})"),
    ):
        ax.axhline(1.0, color=INK_2, linestyle="--", linewidth=1.1)
        ax.plot(x, prof_low[idx], color=BLUE, linewidth=1.8, marker="o",
                markersize=4, label="low margin")
        ax.plot(x, prof_high[idx], color=ORANGE, linewidth=1.8, marker="s",
                markersize=4, label="high margin")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        style(ax, title=title)
        legend(ax)

    ax_init.set_ylabel(r"$\|\delta^l\| / \|\delta^{l-1}\|$", color=INK_2, fontsize=9.5)
    save(fig, save_path)


def plot_directional(epochs, results, save_path_prefix="img/directional"):
    """
    Expansion along random / boundary-crossing / most-amplified directions.

    The gap between the random and margin series is the point: random probes in
    784 dimensions barely touch the subspace that decides classifications.
    """
    styles = {
        "random": (ORANGE, "o", "-"),
        "margin": (BLUE, "s", "-"),
        "top": (AQUA, "^", "--"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (name, r) in zip(axes, results.items()):
        for key, (color, marker, ls) in styles.items():
            ax.plot(epochs, r[key], color=color, marker=marker, linestyle=ls,
                    linewidth=1.8, markersize=4, label=key)
        ax.set_yscale("log")
        style(ax, title=name, xlabel="epoch")
        legend(ax)
    axes[0].set_ylabel("cumulative expansion", color=INK_2, fontsize=9.5)
    save(fig, f"{save_path_prefix}_expansion.png")

    fig, (ax_conc, ax_grp) = plt.subplots(1, 2, figsize=(13, 4.8))

    for (name, r), color in zip(results.items(), (BLUE, ORANGE)):
        conc = np.array(r["margin"]) / np.maximum(np.array(r["random"]), 1e-30)
        ax_conc.plot(epochs, conc, color=color, linewidth=1.8, marker="o",
                     markersize=4, label=name)
    ax_conc.axhline(1.0, color=INK_2, linestyle="--", linewidth=1.1)
    style(ax_conc, title="Sensitivity concentration  (1.0 = boundary direction is unremarkable)",
          xlabel="epoch", ylabel="margin direction / random direction")
    legend(ax_conc)

    low, high = results["low margin"], results["high margin"]
    ax_grp.axhline(1.0, color=INK_2, linestyle="--", linewidth=1.1)
    for key, (color, marker, ls) in styles.items():
        ratio = np.array(low[key]) / np.maximum(np.array(high[key]), 1e-30)
        ax_grp.plot(epochs, ratio, color=color, marker=marker, linestyle=ls,
                    linewidth=1.8, markersize=4, label=key)
    style(ax_grp, title="Geometric differentiation, per direction type",
          xlabel="epoch", ylabel="low margin / high margin")
    legend(ax_grp)

    save(fig, f"{save_path_prefix}_ratio.png")
