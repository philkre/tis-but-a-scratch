"""Training-run figures: loss/accuracy curves and the confusion matrix."""

import numpy as np
import matplotlib.pyplot as plt

from viz.style import BLUE, GRID, INK, INK_2, ORANGE, legend, save, style


def plot_history(history, save_path="img/training_curves.png"):
    """Loss and accuracy, train vs val, to spot under/overfitting."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, key_train, key_val, label in (
        (ax_loss, "train_loss", "val_loss", "cross-entropy loss"),
        (ax_acc, "train_acc", "val_acc", "accuracy"),
    ):
        ax.plot(epochs, history[key_train], color=BLUE, linewidth=1.8, label="train")
        ax.plot(epochs, history[key_val], color=ORANGE, linewidth=1.8, label="validation")
        style(ax, title=label.capitalize(), xlabel="epoch", ylabel=label)
        legend(ax)

    save(fig, save_path)


def plot_confusion_matrix(cm, save_path="img/confusion_matrix.png"):
    """Counts per (true, predicted) pair, with the diagonal visible by contrast."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    style(ax, title="Confusion matrix",
          xlabel="predicted label", ylabel="true label")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    cbar.ax.tick_params(colors=INK_2, labelsize=8)

    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7.5,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)

    save(fig, save_path)
