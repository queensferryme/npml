from io import StringIO

import numpy as np

from npml.cluster.kmeans import KMeans


def test_kmeans(k: int = 3):
    model = KMeans(k)
    data = np.genfromtxt("assets/data.csv", delimiter=",")[:, 1:]
    cluster, k_mean_vectors = model.fit(data)
    assert cluster.shape == (data.shape[0],)
    assert k_mean_vectors.shape == (k, data.shape[1])
