import os
import numpy as np
import matplotlib.pyplot as plt


def confusion_matrix(model, X, y, batch_size=256):
    """
    Forward pass, argmax predictions vs argmax(y).
    Returns a (10, 10) int ndarray, rows = true label, cols = predicted label.
    """
    cm = np.zeros((10, 10), dtype=int)
    for start in range(0, X.shape[0], batch_size):
        xb = X[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model.forward(xb)
        preds = np.argmax(logits, axis=1)
        labels = np.argmax(yb, axis=1)
        np.add.at(cm, (labels, preds), 1)
    return cm


def plot_confusion_matrix(cm, save_path="img/confusion_matrix.png"):
    """matplotlib imshow of the confusion matrix, saved to save_path."""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)

    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    fig.tight_layout()
    # check if directory exists, create if not
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
