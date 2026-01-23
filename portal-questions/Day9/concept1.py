import pandas as pd
import os ,sys
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def evalute_clusters(x , lables):
    score = silhouette_score(x , lables)
    
    print(f"\nCluster Evaluation:")
    print(f"Silhouette Score: {score:.3f}")
    
    print(f"\nCluster Member Counts:")
    print(pd.Series(lables).value_counts())


df = pd.read_csv(os.path.join(sys.path[0] , input()))
print("Enter your data file (CSV or XLSX): ")
print("Dataset Loaded Successfully!")
print(df.head())
print(df.info())

input_df = df.drop(columns = 'weight_condition_n')
print(f"\nInput Data:")
print(input_df.head())

print(f"\nOutput Data:")
print(df['weight_condition_n'].head())

scalar = StandardScaler().fit_transform(input_df)
print(f"\nScaling Input Data...")

print(f"\nSilhouette Scores WITHOUT PCA:")
for k in range(2 , 10):
    kmeans = KMeans(n_clusters = k , random_state = 10)
    lables = kmeans.fit_predict(scalar)
    
    score = silhouette_score(scalar , lables)
    print(f"k={k}: Silhouette Score = {round(score , 3)}")

print(f"\nRunning KMeans WITHOUT PCA...")

kmeans_final = KMeans(n_clusters = 2 , random_state = 10)
lables_final = kmeans_final.fit_predict(scalar)

evalute_clusters(scalar , lables_final)

print("\nRunning PCA (n_components=2)...")

pca = PCA(n_components = 2)
x_pca = pca.fit_transform(scalar)

print("\nSilhouette Scores WITH PCA:")

for k in range(2 , 10):
    kmeans_pca = KMeans(n_clusters = k , random_state = 10)
    lables_pca = kmeans_pca.fit_predict(x_pca)
    
    score = silhouette_score(x_pca , lables_pca)
    print(f"k={k}: Silhouette Score = {round(score , 3)}")
    
print(f"\nRunning KMeans WITH PCA...")

kmeans_final_pca = KMeans(n_clusters = 2 , random_state = 10)
lables_final_pca = kmeans_final.fit_predict(x_pca)

evalute_clusters(x_pca , lables_final_pca)

print("\n==================== SUMMARY ====================")
print("Data Loaded")
print("Data Scaled")
print("Optimal k checked using Silhouette Score")
print("K-Means applied WITHOUT PCA")
print("K-Means applied WITH PCA")
print("Evaluation completed using silhouette score")
print("==================================================")




