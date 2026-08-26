import os
import numpy as np


def save_checkpoint(model, path):
    flat = {}
    for i, layer in enumerate(model.layers):
        for name, param in layer.params.items():
            flat[f"layer{i}_{name}"] = param
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **flat)


def load_checkpoint(model, path):
    data = np.load(path)
    for i, layer in enumerate(model.layers):
        for name in layer.params:
            layer.params[name][...] = data[f"layer{i}_{name}"]


def accuracy(logits, y_onehot):
    preds = np.argmax(logits, axis=1)
    labels = np.argmax(y_onehot, axis=1)
    return np.mean(preds == labels)


def evaluate(model, loss_layer, X, y, batch_size=256):
    losses = []
    accs = []
    for start in range(0, X.shape[0], batch_size):
        xb = X[start:start + batch_size]
        yb = y[start:start + batch_size]
        logits = model.forward(xb)
        losses.append(loss_layer.forward(logits, yb))
        accs.append(accuracy(logits, yb))
    return np.mean(losses), np.mean(accs)


def train(
    model,
    loss_layer,
    optimizer,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=10,
    batch_size=32,
    checkpoint_path="models/best_model.npz",
    seed=1004,
    epoch_callback=None,
):
    rng = np.random.default_rng(seed)
    n_train = X_train.shape[0]

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        idx = rng.permutation(n_train)
        X_shuf, y_shuf = X_train[idx], y_train[idx]

        epoch_losses = []
        epoch_accs = []

        for start in range(0, n_train, batch_size):
            xb = X_shuf[start:start + batch_size]
            yb = y_shuf[start:start + batch_size]

            logits = model.forward(xb)
            loss = loss_layer.forward(logits, yb)
            dlogits = loss_layer.backward()
            model.backward(dlogits)
            optimizer.step(model)

            epoch_losses.append(loss)
            epoch_accs.append(accuracy(logits, yb))

        train_loss = np.mean(epoch_losses)
        train_acc = np.mean(epoch_accs)
        val_loss, val_acc = evaluate(model, loss_layer, X_val, y_val)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"epoch {epoch:3d}  train_loss {train_loss:.4f}  train_acc {train_acc:.4f}  "
            f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, checkpoint_path)

        if epoch_callback is not None:
            epoch_callback(epoch, model)

    return history
