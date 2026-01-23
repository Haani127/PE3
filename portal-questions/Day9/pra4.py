# main.py

import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")

# Import custom ML evaluation module
import ML_Modules as mm

def load_file(file_path):
    """Automatically detect Excel or CSV."""
    ext = file_path.split(".")[-1].lower()
    if ext in ["xlsx", "xls"]:
        return pd.read_excel(file_path)
    elif ext == "csv":
        return pd.read_csv(file_path)
    else:
        print("Error: Unsupported file format. Please use CSV or Excel.")
        sys.exit(1)

def main():
    # ============================
    # Step 0: Get dataset filename
    # ============================
    filename = input("Enter your dataset filename (CSV or Excel): ").strip()
    file_path = os.path.join(sys.path[0], filename)

    # ============================
    # Step 1: Load dataset
    # ============================
    try:
        df = load_file(file_path)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    print("\n========== FIRST 5 ROWS ==========")
    print(df.head())

    print("\n========== DATASET SHAPE ==========")
    print(df.shape)

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    # ============================
    # Step 2: Select numeric columns
    # ============================
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.empty:
        print("Error: No numeric columns found.")
        sys.exit(1)

    print("\n========== NUMERIC COLUMNS ==========")
    print(df_numeric.columns.tolist())

    X = df_numeric.values

    # ============================
    # Step 3: Silhouette Scores for K range
    # ============================
    print("\n========== SILHOUETTE SCORES ==========")
    for k in range(2, 10):
        model = KMeans(n_clusters=k, random_state=10)
        labels = model.fit_predict(X)
        score = round(mm.silhouette_score(X, labels), 3)
        print(f"k = {k}: Silhouette Score = {score}")

    # ============================
    # Step 4: Build final KMeans Model (k=2)
    # ============================
    print("\n========== FINAL CLUSTER MODEL (k=2) ==========")
    final_k = 2
    kmeans = KMeans(n_clusters=final_k, random_state=10)
    cluster_labels = kmeans.fit_predict(X)
    print("Cluster Labels:")
    print(cluster_labels)

    # ============================
    # Step 5: Evaluate clustering
    # ============================
    print("\n========== CLUSTER EVALUATION ==========")
    mm.evaluate_clusterer(X, cluster_labels)

if __name__ == "__main__":
    main()


#--------------------ML_Modules.py--------------------

# ML_Modules.py

import numpy as np
from sklearn.metrics import (
    silhouette_score as sk_silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# --------------------------------------------------
# Silhouette Score Wrapper
# --------------------------------------------------
def silhouette_score(X, labels):
    """
    Compute Silhouette Score for clustering.

    Parameters:
    X : array-like, shape (n_samples, n_features)
    labels : array-like, shape (n_samples,)

    Returns:
    float
    """
    return sk_silhouette_score(X, labels)


# --------------------------------------------------
# Cluster Evaluation Metrics
# --------------------------------------------------
def evaluate_clusterer(X, labels):
    """
    Evaluate clustering quality using multiple metrics.

    Metrics:
    - Silhouette Score
    - Calinski-Harabasz Index
    - Davies-Bouldin Index
    """

    sil_score = sk_silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    print("Cluster Evaluation Metrics:")
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Calinski-Harabasz Index: {ch_score:.4f}")
    print(f"Davies-Bouldin Index: {db_score:.4f}")
