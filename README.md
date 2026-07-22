## Building Interpretability Intuition from Scratch

LeNet-ish CNN for MNIST, implemented from scratch in raw NumPy. My goal is deriving and hand coding to build mechanical understanding of CNNs to then later pose interpretability questions on that basis.

### Architecture

```
Input        28×28×1
Conv1        6 filters, 5×5, stride 1     → 24×24×6
ReLU
MaxPool      2×2, stride 2                → 12×12×6
Conv2        16 filters, 5×5, stride 1    → 8×8×16
ReLU
MaxPool      2×2, stride 2                → 4×4×16
Flatten                                   → 256
FC1          256 → 120
ReLU
FC2          120 → 84
ReLU
FC3          84 → 10
Softmax + Cross-Entropy loss
```

### Correctness approach

Every layer is verified two ways before it's trusted:

1. **Numerical gradient checking** — analytical gradients from each layer's hand-derived `backward()` are compared against finite-difference approximations on small random inputs, per layer, in isolation.
2. **Overfit-a-tiny-batch smoke test** — once all layers pass their individual checks, the full wired-up model must be able to drive training accuracy on a handful of images to ~100%, proving the whole graph (forward, backward, optimizer) is integrated correctly before a full training run.

### Progress

- [x] Data pipeline (`data.py`) — MNIST via the `mnist` package, cached locally, normalized, split 55k/5k/10k, one-hot labels
- [x] Numerical gradient-check harness (`gradcheck.py`)
- [x] `Dense` layer, forward + backward
- [x] `ReLU` layer, forward + backward
- [ ] `SoftmaxCrossEntropy` loss layer
- [ ] `Flatten` layer
- [ ] `MaxPool2D` layer
- [ ] `im2col` / `col2im`
- [ ] `Conv2D` layer
- [ ] `Sequential` model container
- [ ] `SGD` optimizer
- [ ] Overfit-a-tiny-batch smoke test
- [ ] Full training loop
- [ ] Evaluation + confusion matrix
