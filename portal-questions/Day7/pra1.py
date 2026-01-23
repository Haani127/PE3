import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter(action='ignore')


# ============================
# Replacement for ML_Modules
# ============================

def assess_outliers(df):
    print("Outlier summary using IQR method:")
    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 - 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        print(f"{col}: {len(outliers)} outliers")


def treat_outliers(df):
    df_copy = df.copy()
    for col in df.columns:
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        # Winsorizing
        df_copy[col] = np.where(df_copy[col] < lower, lower,
                        np.where(df_copy[col] > upper, upper, df_copy[col]))
    return df_copy


def data_scale(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)


# ============================
# MAIN PROGRAM
# ============================

def main():

    # Step 0: get filename
    filename = input("").strip()
    file_path = os.path.join(sys.path[0], filename)

    # Step 1: Load CSV
    try:
        car_sales = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    # Step 2: Show first rows
    print("First 5 rows of the dataset:")
    print(car_sales.head())
    print()

    # Step 3: Info
    print("Dataset information:")
    print(car_sales.info())
    print()

    # Step 4: Missing values
    print("Missing values:")
    print(car_sales.isna().sum())
    print()

    # Step 5: Drop missing rows
    car_sales.dropna(inplace=True)
    print("Rows after removing missing values:", car_sales.shape[0])
    print()

    # Step 6: Numeric columns
    car_sales_numeric = car_sales.select_dtypes(include='number')
    print("Numeric columns:")
    print(list(car_sales_numeric.columns))
    print()

    # Step 7: Outlier assessment
    assess_outliers(car_sales_numeric)
    print()

    # Step 8: Outlier treatment
    treated_df = treat_outliers(car_sales_numeric)
    print("Outliers treated.")
    print()

    # Step 9: Scaling width & length
    if 'Width' in treated_df.columns and 'Length' in treated_df.columns:
        scaled_df = data_scale(treated_df[['Width', 'Length']])
        print("Scaled Width & Length:")
        print(scaled_df.head())
    else:
        print("Width/Length columns not found.")
    print()

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
