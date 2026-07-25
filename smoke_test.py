"""
Load 10 real training images + labels (via data.load_mnist(), just slice
X_train[:10], y_train[:10]).
Build the full model (Task 10) + SoftmaxCrossEntropy + SGD(lr=0.05 or so).
Loop ~200-500 steps: forward -> loss -> backward -> step. Print loss and
train accuracy every ~25 steps.
Assert final train accuracy on these 10 images is >= 0.9 (allow some
slack short of exactly 100%).
"""

import numpy as np

from data import load_mnist
from model import Sequential
from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, SoftmaxCrossEntropy
from optimizer import SGD

x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()
x, y = x_train[:10], y_train[:10]

model = Sequential(
    [
        Conv2D(1, 6, 5),
        ReLU(),
        MaxPool2D(2, 2),
        Conv2D(6, 16, 5),
        ReLU(),
        MaxPool2D(2, 2),
        Flatten(),
        Dense(256, 120),
        ReLU(),
        Dense(120, 84),
        ReLU(),
        Dense(84, 10),
    ]
)
loss_fn = SoftmaxCrossEntropy()
optimizer = SGD(lr=0.05)

n_steps = 300
for step in range(1, n_steps + 1):
    logits = model.forward(x)
    loss = loss_fn.forward(logits, y)
    dlogits = loss_fn.backward()
    model.backward(dlogits)
    optimizer.step(model)

    if step == 1 or step % 25 == 0:
        preds = np.argmax(logits, axis=1)
        labels = np.argmax(y, axis=1)
        acc = np.mean(preds == labels)
        print(f"step {step:4d}  loss {loss:.4f}  train acc {acc:.2f}")

final_logits = model.forward(x)
final_preds = np.argmax(final_logits, axis=1)
final_labels = np.argmax(y, axis=1)
final_acc = np.mean(final_preds == final_labels)

assert (
    final_acc >= 0.9
), f"overfit smoke test failed: final train acc {final_acc:.2f} < 0.9"
print(f"Smoke test passed: overfit accuracy {final_acc:.2f} on 10 images.")
