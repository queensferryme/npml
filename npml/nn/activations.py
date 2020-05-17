import abc

import numpy as np


class Activation(abc.ABC):
    @abc.abstractmethod
    def backward(self, x: np.array) -> np.array:
        pass

    @abc.abstractmethod
    def forward(self, x: np.array) -> np.array:
        pass


class LeakyRelu(Activation):
    """https://arxiv.org/abs/1505.00853"""

    def __init__(self, slope: float) -> None:
        self.slope = slope

    def backward(self, x: np.array) -> np.array:
        return np.where(x > 0, 1.0, self.slope)

    def forward(self, x: np.array) -> np.array:
        return np.maximum(x, self.slope * x)


class Linear(Activation):
    def backward(self, x: np.array) -> np.array:
        return x

    def forward(self, x: np.array) -> np.array:
        return x
