import numpy as np

from npml.nn.activations import Activation
from npml.nn.layers.layer import Layer


class Dense(Layer):
    def __init__(self, n_output: int, *, activation: Activation, **kwargs):
        self.activation = activation
        self.cache, self.params = {}, {}
        self.n_output = n_output
        self.n_input = None if "n_input" not in kwargs else kwargs["n_input"]

    def backward(self, da: np.array) -> np.array:
        assert len(da.shape) == 2 and da.shape[1] == self.n_output
        n_batch = da.shape[0]
        dy = self.activation.backward(da)
        self.cache["dw"] = np.dot(self.cache["x"].T, dy) / n_batch
        self.cache["db"] = np.sum(dy, axis=0, keepdims=True) / n_batch
        return np.dot(dy, self.params["w"].T)

    def forward(self, x: np.array) -> np.array:
        assert len(x.shape) == 2 and x.shape[1] == self.n_input
        self.cache["x"] = x
        y = np.dot(x, self.params["w"]) + self.params["b"]
        return self.activation.forward(y)

    def update(self, lr: float) -> None:
        self.params["b"] -= lr * self.cache["db"]
        self.params["w"] -= lr * self.cache["dw"]
