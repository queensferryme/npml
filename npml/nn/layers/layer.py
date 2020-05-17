import abc

import numpy as np


class Layer(abc.ABC):
    @abc.abstractmethod
    def forward(self, da: np.array) -> np.array:
        pass

    @abc.abstractmethod
    def forward(self, x: np.array) -> np.array:
        pass

    @abc.abstractmethod
    def update(self, lr: float) -> None:
        pass
