"""Figures for the white-box generalization sweep."""

import numpy as np
import matplotlib.pyplot as plt

from viz.style import BLUE, GRID, INK_2, ORANGE, RAMP, legend, save, style


def plot_measure_vs_accuracy(measures, test_acc, noise, taus,
                             save_path="img/generalization_measures.png"):
    """
    One panel per measure: its value against true test accuracy, one point per
    model, colored by noise level. A measure that predicts generalization
    produces a monotone cloud; one that doesn't produces a blob.
    """
    names = list(measures)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2))
    axes = np.atleast_1d(axes)
    levels = sorted(set(noise))

    for ax, name in zip(axes, names):
        for level, color in zip(levels, RAMP):
            m = noise == level
            ax.scatter(measures[name][m], test_acc[m], s=34, color=color,
                       linewidths=0, label=f"{level:.0%} noise")
        style(ax, title=f"{name}\n" + r"Kendall $\tau$ = " + f"{taus[name]:+.2f}",
              xlabel=name.replace("_", " "), ylabel="test accuracy")

    legend(axes[0], fontsize=7.5, loc="best")
    save(fig, save_path)


def plot_measure_trajectories(epochs, per_noise, measure_name,
                              save_path="img/generalization_trajectory.png"):
    """
    A measure's development over training, averaged across seeds, one line per
    noise level. Separation appearing early means the measure predicts rather
    than merely describes.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for (level, values), color in zip(sorted(per_noise.items()), RAMP):
        ax.plot(epochs, values, color=color, linewidth=1.8, label=f"{level:.0%} noise")
    style(ax, title=f"{measure_name} over training",
          xlabel="epoch", ylabel=measure_name.replace("_", " "))
    legend(ax)
    save(fig, save_path)
