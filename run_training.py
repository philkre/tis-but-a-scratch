"""
Full training run: builds the reference model, trains on all 55k images,
evaluates the best checkpoint on the untouched 10k test set, and writes
training curves + confusion matrix to img/.

This is the reference recipe -- everything downstream (bottleneck sweeps,
SAE experiments) should start from a model trained this way.
"""

import numpy as np
import matplotlib.pyplot as plt

from data import load_mnist
from model import Sequential
from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, SoftmaxCrossEntropy
from optimizer import SGD
from train import train, evaluate, load_checkpoint
from evaluate import confusion_matrix, plot_confusion_matrix


def build_model(seed=1004):
    """
    Reference architecture. Each weighted layer gets its own distinct seed
    derived from one master seed -- Conv2D/Dense both default to seed=1004,
    so without this every layer would draw from the *same* RNG stream and
    their weights would be correlated across layers.
    """
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2**31 - 1, size=5)

    return Sequential(
        [
            Conv2D(1, 6, 5, seed=int(s[0])),
            ReLU(),
            MaxPool2D(2, 2),
            Conv2D(6, 16, 5, seed=int(s[1])),
            ReLU(),
            MaxPool2D(2, 2),
            Flatten(),
            Dense(256, 120, seed=int(s[2])),
            ReLU(),
            Dense(120, 84, seed=int(s[3])),
            ReLU(),
            Dense(84, 10, seed=int(s[4])),
        ]
    )


def plot_history(history, save_path="img/training_curves.png"):
    """Loss and accuracy curves, train vs val, to spot under/overfitting."""
    import os

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="val")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    LR = 0.1
    EPOCHS = 20
    BATCH_SIZE = 32
    CHECKPOINT = "models/best_model.npz"

    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()
    print(f"train {x_train.shape[0]}  val {x_val.shape[0]}  test {x_test.shape[0]}")

    model = build_model()
    loss_fn = SoftmaxCrossEntropy()
    optimizer = SGD(lr=LR)

    print(f"training: lr={LR}  epochs={EPOCHS}  batch_size={BATCH_SIZE}")
    history = train(
        model,
        loss_fn,
        optimizer,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        checkpoint_path=CHECKPOINT,
    )

    plot_history(history)

    # evaluate the BEST checkpoint (not the final epoch's weights) on the
    # untouched test set
    load_checkpoint(model, CHECKPOINT)
    test_loss, test_acc = evaluate(model, loss_fn, x_test, y_test)
    print(f"\nbest checkpoint on test set: loss {test_loss:.4f}  acc {test_acc:.4f}")

    cm = confusion_matrix(model, x_test, y_test)
    plot_confusion_matrix(cm)
    print(cm)
    print("\nwrote img/training_curves.png, img/confusion_matrix.png")
