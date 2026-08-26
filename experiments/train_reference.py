"""
The reference training run: full 55k training set, best checkpoint kept.

Everything downstream starts from a model trained this way.

    python -m experiments.train_reference
"""

from nn import SGD, SoftmaxCrossEntropy, build_model, evaluate, load_checkpoint, load_mnist, train
from analysis.metrics import confusion_matrix
from viz.training import plot_confusion_matrix, plot_history

LR = 0.1
EPOCHS = 20
BATCH_SIZE = 32
CHECKPOINT = "models/best_model.npz"


def main():
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()
    print(f"train {x_train.shape[0]}  val {x_val.shape[0]}  test {x_test.shape[0]}")

    model = build_model()
    loss_fn = SoftmaxCrossEntropy()

    print(f"training: lr={LR}  epochs={EPOCHS}  batch_size={BATCH_SIZE}")
    history = train(
        model, loss_fn, SGD(lr=LR),
        x_train, y_train, x_val, y_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE, checkpoint_path=CHECKPOINT,
    )
    plot_history(history)

    # the best checkpoint, not the final epoch's weights
    load_checkpoint(model, CHECKPOINT)
    test_loss, test_acc = evaluate(model, loss_fn, x_test, y_test)
    print(f"\nbest checkpoint on test set: loss {test_loss:.4f}  acc {test_acc:.4f}")

    cm = confusion_matrix(model, x_test, y_test)
    plot_confusion_matrix(cm)
    print(cm)
    print("\nwrote img/training_curves.png, img/confusion_matrix.png")


if __name__ == "__main__":
    main()
