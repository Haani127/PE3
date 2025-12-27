# main.py

import warnings
warnings.simplefilter(action='ignore')
import os,sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    filename = input().strip()          # e.g., Sample.csv

    # Basic filename / existence check
    if not filename.endswith(".csv"):
        print(f"Error: File '{filename}' not found.")
        return

    # try:
    df = pd.read_csv(os.path.join(sys.path[0],filename))


    # Required columns
    required_cols = [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_montly_hours",
        "time_spend_company",
        "Work_accident",
        "left",
        "promotion_last_5years",
        "Department",
        "salary",
    ]
    if not all(col in df.columns for col in required_cols):
        print("Error: Input file does not contain all required columns.")
        return

    # 1. First 5 rows of the dataset
    print("First 5 rows of the dataset:")
    head_df = df.head()
    print(head_df)

    # 2. Number of samples in the data
    print("\nNumber of samples in the data:")
    print(df.shape[0])
    print()

    # 3. Data types of each column
    print("Data types of each column:")
    print(df.dtypes)
    print()

    # 4. Feature columns used for classification
    feature_cols = [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_montly_hours",
        "time_spend_company",
        "Work_accident",
        "promotion_last_5years",
    ]
    print("Feature columns:")
    print(feature_cols)
    print()



    # 5. Statistical summary of numeric columns
    print("Statistical summary of numeric columns:")
    desc = df.describe()  
    print(desc)

    X = df[feature_cols]
    y = df["left"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = DecisionTreeClassifier(
        random_state=42,
        max_depth=4,
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\nModel Accuracy: {acc}")
    print()
    print("Classification Report:")
    print(report)


if __name__ == "__main__":
    main()