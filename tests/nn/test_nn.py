import numpy as np

from npml.nn.activations import LeakyRelu, Linear
from npml.nn.layers import Dense
from npml.nn.model import Model


def test_nn():
    data = np.genfromtxt("assets/boston.csv")
    data = data / np.max(data, axis=0)
    model = Model(
        [
            Dense(5, activation=LeakyRelu(0.1), n_input=data.shape[1] - 1),
            Dense(1, activation=Linear()),
        ]
    )
    model.compile()
    model.fit(data[:, :-1], data[:, -1:], epochs=10, lr=0.1)
