import os
import sys
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.simplefilter(action='ignore')


def main():
    # ============================
    # Step 0: Get CSV filename
    # ============================
    filename = input("Enter your dataset filename (CSV): ").strip()
    file_path = os.path.join(sys.path[0], filename)

    # ============================
    # Step 1: Load CSV File
    # ============================
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    # ============================
    # Step 2: Display first 5 rows
    # ============================
    print("First 5 rows of the dataset:")
    print(df.head())
    print()

    # ============================
    # Step 3: Display dataset shape
    # ============================
    print("Number of samples and features:")
    print(df.shape)
    print()

    # ============================
    # Step 4: Display data types
    # ============================
    print("Data types of each column:")
    print(df.dtypes)
    print()

    # ============================
    # Step 5: Select numeric columns for clustering
    # ============================
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.empty:
        print("Error: No numeric columns found for clustering.")
        sys.exit(1)

    print("Numeric columns used for clustering:")
    print(df_numeric.columns.tolist())
    print()

    X = df_numeric.values

    # ============================
    # Step 6: Dimensionality Reduction using PCA
    # ============================
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(X)
    print("Original shape:", X.shape)
    print("Reduced shape after PCA:", reduced_data.shape)
    print()

    # ============================
    # Step 7: Find best number of clusters using Silhouette Score
    # ============================
    print("Silhouette Scores for different cluster sizes:")
    for k in range(2, 10):
        temp_model = KMeans(n_clusters=k, random_state=10)
        cluster_labels = temp_model.fit_predict(reduced_data)
        score = round(silhouette_score(reduced_data, cluster_labels), 3)
        print(f"For n_clusters = {k}, Average Silhouette Score = {score}")
    print()

    # ============================
    # Step 8: Build KMeans Model with reduced dimensions
    # ============================
    final_k = 3
    kmeans_dr = KMeans(n_clusters=final_k, random_state=10)
    cluster_labels = kmeans_dr.fit_predict(reduced_data)
    print(f"Cluster labels assigned (K={final_k}):")
    print(cluster_labels)
    print()

    # ============================
    # Step 9: Optional - Evaluation of clustering
    # ============================
    try:
        import ML_Modules as mm
        mm.evaluate_clusterer(reduced_data, cluster_labels)
    except ModuleNotFoundError:
        print("ML_Modules not found. Skipping evaluation.")
    except AttributeError:
        print("evaluate_clusterer function not found in ML_Modules. Skipping evaluation.")


if __name__ == "__main__":
    main()
#--------------------ML_Modules.py--------------------


# ML_Modules.py
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_clusterer(X, labels):
    """
    Evaluate clustering results using standard metrics.
    """
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    
    print("\n========== Cluster Evaluation Metrics ==========")
    print(f"Silhouette Score: {sil:.4f}")
    print(f"Calinski-Harabasz Index: {ch:.4f}")
    print(f"Davies-Bouldin Index: {db:.4f}")
# ML_Modules.py
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate_clusterer(X, labels):
    """
    Evaluate clustering results using standard metrics.
    """
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    
    print("\n========== Cluster Evaluation Metrics ==========")
    print(f"Silhouette Score: {sil:.4f}")
    print(f"Calinski-Harabasz Index: {ch:.4f}")
    print(f"Davies-Bouldin Index: {db:.4f}")
