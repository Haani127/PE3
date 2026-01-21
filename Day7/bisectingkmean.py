from sklearn.cluster import BisectingKMeans
import numpy as np

x = np.array([
    [1, 2],
    [2, 1],
    [3, 2],
    [8, 8],
    [9, 8],
    [8, 9]
])

model = BisectingKMeans(
    n_clusters=2,
    random_state=42
).fit(x)

labels = model.labels_
print("Cluster labels:", labels)
