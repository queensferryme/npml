from io import StringIO

import numpy as np

from npml.cluster.kmeans import KMeans

# 周志华《机器学习》西瓜数据集4.0
DATA = """
1,0.697,0.460
2,0.774,0.376
3, 0.634,0.264
4,0.608,0.318
5,0.556,0.215
6,0.403,0.237
7,0.481,0.149
7,0.666,0.091
8,0.437,0.211
9,0.666,0.091
10,0.243,0.267
11,0.245,0.057
12,0.343,0.099
13,0.639,0.161
14,0.657,0.198
15,0.360,0.370
16,0.593,0.042
17,0.719,0.103
18,0.359,0.188
19,0.339,0.241
20,0.282,0.257
21,0.748,0.232
22,0.714,0.346
23,0.483,0.312
24,0.478,0.437
25,0.525,0.369
26,0.751,0.489
27,0.532,0.472
28,0.473,0.376
29,0.725,0.445
30,0.446,0.459
"""


def test_kmeans(k: int = 3):
    model = KMeans(k)
    data = np.genfromtxt(StringIO(DATA), delimiter=",")[:, 1:]
    cluster, k_mean_vectors = model.fit(data)
    assert cluster.shape == (data.shape[0],)
    assert k_mean_vectors.shape == (k, data.shape[1])
