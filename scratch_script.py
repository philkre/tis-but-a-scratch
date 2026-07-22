from data import load_mnist
from gradcheck import check_dense

x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()
assert x_train.shape == (55000, 1, 28, 28)
assert y_train.shape == (55000, 10)
assert x_train.min() >= 0.0 and x_train.max() <= 1.0
assert y_train.min() >= 0.0 and y_train.max() <= 1.0

print("Data loaded successfully.")

check_dense()
