import os
import sys

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(os.path.join(sys.path[0], input()))
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound).round(2)

df = df.drop(columns=['Detergents_Paper'])
scaled_df = StandardScaler().fit_transform(df)

dbscan = DBSCAN(eps=2, min_samples=5)
labels = dbscan.fit_predict(scaled_df)

silhouette_score = silhouette_score(scaled_df, labels)
print(f"The average silhouette_score is: {silhouette_score:.2f}")

calinski = calinski_harabasz_score(scaled_df, labels)
print(f"Calinski-Harabasz Index: {calinski:.2f}")

davis = davies_bouldin_score(scaled_df, labels)
print(f"Davies-Bouldin Index: {davis:.2f}")