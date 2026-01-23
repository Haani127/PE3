import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
warnings.filterwarnings("ignore")
def main():
    filename = input().strip()
    df = pd.read_csv(os.path.join(sys.path[0], filename))
    print(df.head())
    df.info()
    print()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(X_scaled)
    distances, indices = neighbors_fit.kneighbors(X_scaled)
    distances = np.sort(distances[:, 1], axis=0) 
    eps_val = 2
    for min_samples in [3, 4, 5]:
        dbscan = DBSCAN(eps=eps_val, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)
        transformed_labels = labels + 2
        unique_labels, counts = np.unique(transformed_labels, return_counts=True)
        results = list(zip(unique_labels, counts))
        print(f"eps= {eps_val} | min_samples=  {min_samples} | obtained clustering:  {results}")
    print()
    final_eps = 2
    final_min_samples = 3
    dbscan_final = DBSCAN(eps=final_eps, min_samples=final_min_samples)
    final_labels = dbscan_final.fit_predict(X_scaled)
    cluster_counts = pd.Series(final_labels).value_counts().sort_values(ascending=False)
    cluster_counts.name = "count"
    print("cluster")
    print(cluster_counts.to_string())
    print(f"Name: {cluster_counts.name}, dtype: {cluster_counts.dtype}\n")
    unique_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
    if len(set(final_labels)) > 1:
        sil_score = silhouette_score(X_scaled, final_labels)
        print(f"The average silhouette_score is: {sil_score:.2f}\n")
    else:
        print("The average silhouette_score is: 0.00\n")
    if unique_clusters >= 1 and len(set(final_labels)) > 1:
        ch_score = calinski_harabasz_score(X_scaled, final_labels)
        print(f"Calinski-Harabasz Index: {ch_score:.2f}\n")
    else:
        print("Calinski-Harabasz Index: 0.00\n")
    if unique_clusters >= 1 and len(set(final_labels)) > 1:
        db_index = davies_bouldin_score(X_scaled, final_labels)
        print(f"Davies-Bouldin Index: {db_index:.2f}")
    else:
        print("Davies-Bouldin Index: 0.00")
if __name__ == "__main__":
    main()