from data import load_mnist
from gradcheck import check_all
from model import Sequential
from layers import *
from train import train
from optimizer import SGD
from evaluate import *

x_train, y_train, x_val, y_val, x_test, y_test = load_mnist()
assert x_train.shape == (55000, 1, 28, 28)
assert y_train.shape == (55000, 10)
assert x_train.min() >= 0.0 and x_train.max() <= 1.0
assert y_train.min() >= 0.0 and y_train.max() <= 1.0

print("Data loaded successfully.")

check_all()

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
optimizer = SGD(lr=0.01)

train(model, loss_fn, optimizer, x_train[:2000], y_train[:2000], x_val, y_val, epochs=3)

cm = confusion_matrix(model, x_val, y_val)
print(cm)
plot_confusion_matrix(cm)
