import numpy as np
from sklearn.cluster import AgglomerativeClustering

X = np.array([
    [1, 2],
    [2, 1],
    [3, 2],
    [8, 8],
    [9, 8],
    [8, 9]
])

model = AgglomerativeClustering(
    n_clusters=2,
    linkage='ward'
)

labels = model.fit_predict(X)

print("Cluster labels:", labels)
