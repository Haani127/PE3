import pandas as pd
import numpy as np
import os, sys

from sklearn.cluster import DBSCAN
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

for min_neighbours in [3, 4, 5]:
    dbscan = DBSCAN(eps=2, min_samples=min_neighbours)
    labels = dbscan.fit_predict(scaled_df)
    transformed_labels = labels + 2
    unique_labels, counts = np.unique(transformed_labels, return_counts=True)
    results = list(zip(unique_labels, counts))

    print(f"eps= {2} | min_samples=  {min_neighbours} | obtained clustering:  {results}")