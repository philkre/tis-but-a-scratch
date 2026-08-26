"""
Integration check: the full wired model must overfit 10 images to ~100%.

Every layer passes its own gradient check, so a failure here is integration --
layer order, an optimizer sign error, a shape mismatch numpy broadcast away.

    python -m experiments.smoke_test
"""

import numpy as np

from nn import SGD, SoftmaxCrossEntropy, build_model, load_mnist

N_IMAGES = 10
N_STEPS = 300
LR = 0.05


def main():
    x_train, y_train, *_ = load_mnist()
    x, y = x_train[:N_IMAGES], y_train[:N_IMAGES]

    model = build_model()
    loss_fn = SoftmaxCrossEntropy()
    optimizer = SGD(lr=LR)

    for step in range(1, N_STEPS + 1):
        logits = model.forward(x)
        loss = loss_fn.forward(logits, y)
        model.backward(loss_fn.backward())
        optimizer.step(model)

        if step == 1 or step % 25 == 0:
            acc = np.mean(np.argmax(logits, axis=1) == np.argmax(y, axis=1))
            print(f"step {step:4d}  loss {loss:.4f}  train acc {acc:.2f}")

    final = model.forward(x)
    acc = np.mean(np.argmax(final, axis=1) == np.argmax(y, axis=1))
    assert acc >= 0.9, f"overfit smoke test failed: final train acc {acc:.2f} < 0.9"
    print(f"Smoke test passed: overfit accuracy {acc:.2f} on {N_IMAGES} images.")


if __name__ == "__main__":
    main()
