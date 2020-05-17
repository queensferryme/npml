from typing import List

import numpy as np

from npml.nn.layers import Layer


class Model:
    def __init__(self, layers: List[Layer]):
        assert len(layers)
        self.layers = layers

    def backward(self, da: np.array) -> np.array:
        for layer in reversed(self.layers):
            da = layer.backward(da)

    def compile(self) -> None:
        for prev_layer, next_layer in zip(self.layers[:-1], self.layers[1:]):
            next_layer.n_input = prev_layer.n_output
        for layer in self.layers:
            # TODO: implement various initialization methods
            layer.params["b"] = np.random.randn(1, layer.n_output)
            layer.params["w"] = np.random.randn(layer.n_input, layer.n_output)

    def fit(self, x: np.array, y: np.array, *, epochs: int, lr: float) -> None:
        for _ in range(epochs):
            y_hat = self.forward(x)
            self.backward(y_hat - y)
            self.update(lr)
            # TODO: implement various loss functions
            print("mse:", np.sum((y_hat - y) ** 2) / x.shape[0])

    def forward(self, x: np.array) -> np.array:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def update(self, lr: float) -> None:
        for layer in self.layers:
            layer.update(lr)
