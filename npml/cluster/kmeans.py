import numpy as np

from typing import Tuple


class KMeans:
    def __init__(self, k: int):
        self.k = k

    def fit(
        self, data: np.array, *, epochs: int = 100, tolerance: float = 1e-4
    ) -> Tuple[np.array, np.array]:
        cluster = np.zeros((data.shape[0],))
        k_mean_vectors = data[np.random.randint(data.shape[0], size=self.k)]
        for _ in range(epochs):
            distances = np.hstack(
                [
                    np.sqrt(np.sum((vector - data) ** 2, axis=1, keepdims=True))
                    for vector in k_mean_vectors
                ]
            )
            cluster = np.argmin(distances, axis=1)
            new_k_mean_vectors = np.vstack(
                [np.average(data[cluster == i], axis=0) for i in range(self.k)]
            )
            if np.all((new_k_mean_vectors - k_mean_vectors) < tolerance):
                return cluster, k_mean_vectors
            k_mean_vectors = new_k_mean_vectors
        return cluster, k_mean_vectors
