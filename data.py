import numpy as np
import mnist
import os

DATA_DIR = "data_cache"


def load_mnist(seed=1004):
    """
    Returns (x_train, y_train, x_val, y_val, x_test, y_test).

    x_*: float64 ndarray, shape (N, 1, 28, 28), values in [0, 1]
    y_*: float64 ndarray, shape (N, 10), values in [0, 9], one hot encoded

    Split: 55k train / 5k val from standard 60k training set
    Standard 10k test set
    """

    # check dir
    os.makedirs(DATA_DIR, exist_ok=True)
    mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    mnist.temporary_dir = lambda: DATA_DIR

    # load images
    x_train, y_train = mnist.train_images(), mnist.train_labels()
    x_test, y_test = mnist.test_images(), mnist.test_labels()

    # normalise images
    x_train = x_train.astype(np.float64) / 255.0
    x_test = x_test.astype(np.float64) / 255.0

    # reshape images to (N, 1, 28, 28)
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)

    # one hot encode labels
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    # val split (randomly select 5k from training set)
    rng = np.random.default_rng(seed)
    val_idx = rng.choice(x_train.shape[0], 5000, replace=False)
    x_val = x_train[val_idx]
    y_val = y_train[val_idx]
    x_train = np.delete(x_train, val_idx, axis=0)
    y_train = np.delete(y_train, val_idx, axis=0)

    return x_train, y_train, x_val, y_val, x_test, y_test
