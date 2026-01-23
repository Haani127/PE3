import pandas as pd
import numpy as np
import os, sys
from sklearn.preprocessing import StandardScaler
def assess_outliers():
    pass
def main():
    filename = input().strip()
    df = pd.read_csv(os.path.join(sys.path[0], filename))
    print("Dataset Preview:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nDataset Description:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    numeric_df = df.select_dtypes(include="number")
    for col in numeric_df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    df = df.round(2)
    for col in df.columns:
        df[col] = df[col].astype(float)
    print("\nData After Outlier Treatment:")
    print(df.head())
    print("\nMulticollinearity Matrix:")
    corr_matrix = df.corr().abs()
    bool_matrix = corr_matrix >= 0.7
    print(bool_matrix)
    columns_to_drop = ['Detergents_Paper']
    df_reduced = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    print("\nColumns after removal:")
    print(df_reduced.columns.tolist())
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_reduced)
    df_scaled = pd.DataFrame(scaled_data, columns=df_reduced.columns)
    print("\nScaled Data Preview:")
    print(df_scaled.head())
if __name__ == "__main__":
    main()