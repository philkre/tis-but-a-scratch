"""
Single source of truth for the reference architecture.

Before this existed the same twelve-layer list was hardcoded in five separate
files, so changing the network meant editing all five.
"""

import numpy as np

from nn.layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense
from nn.model import Sequential

N_CLASSES = 10
INPUT_SHAPE = (1, 28, 28)


def build_model(seed=1004):
    """
    LeNet-ish reference model.

    Each weighted layer gets its own seed derived from one master seed.
    Conv2D and Dense both default to seed=1004, so without this every layer
    would draw from the same RNG stream and their weights would be correlated
    across layers.
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
