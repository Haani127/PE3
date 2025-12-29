import pandas as pd
import numpy as np
import os
import sys
import ML_Modules as mm
from sklearn.svm import SVC

def main():

    filename = input("").strip()
    file_path = os.path.join(sys.path[0], filename)


    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)

    print("# Head of all columns")
    print(df.head())
    print()

    print("# Data Types of all columns")
    print(df.dtypes)
    print()

    diab_df = df[['Glucose', 'BMI', 'Age', 'FamilyHistory', 'HbA1c', 'Outcome']]
    
    print("# Working subset head")
    print(diab_df.head())
    print()

    print("# Mean values grouped by Outcome")
    print(diab_df.groupby('Outcome').mean())
    print()

    print("# Null value check")
    print(diab_df.isnull().sum())
    print()

    print("# Zero-value count for BMI")
    print((diab_df['BMI'] == 0).sum())
    print()

    print("# Zero-value count for Glucose")
    print((diab_df['Glucose'] == 0).sum())
    print()

    print("# Zero-value count for Age")
    print((diab_df['Age'] == 0).sum())
    print()

    diab_df = diab_df[(diab_df['Glucose'] != 0) & (diab_df['BMI'] != 0)]

    print("# Zero-value count after removal: Glucose")
    print((diab_df['Glucose'] == 0).sum())
    print()

    print("# Zero-value count after removal: BMI")
    print((diab_df['BMI'] == 0).sum())
    print()

    print("# Number of rows after zero-value removal")
    print(diab_df.shape[0])
    print()

    mm.assess_outliers(diab_df.iloc[:, :-1])

    diab_treated_df = mm.treat_outliers(diab_df.iloc[:, :-1])
    print("# Data after outlier treatment")
    print(diab_treated_df)
    print()


    X_DT = diab_treated_df                     
    y_DT = diab_df['Outcome']                  

    X_scaled = mm.data_scale(X_DT)

    X_train, X_test, y_train, y_test = mm.split_data(X_scaled, y_DT, size=0.2)

    svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)

    print("# SVM Model Evaluation")
    mm.evaluate_classifier(y_test, y_pred)


if __name__ == "__main__":
    main()


#---------------ML_MOdules.py---------------


import pandas as pd
import numpy as np


def assess_outliers(data1):
    return None


def treat_outliers(data1):
    def iqr_winsorization(df, thresh=1.5):
        df_out = df.copy()
        columns_to_treat = df_out.columns

        for col in columns_to_treat:
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_fence = Q1 - thresh * IQR
            upper_fence = Q3 + thresh * IQR

            df_out.loc[df_out[col] < lower_fence, col] = lower_fence
            df_out.loc[df_out[col] > upper_fence, col] = upper_fence

        return df_out

    df2 = pd.DataFrame(data1)
    treated_df = iqr_winsorization(df2.copy())
    return treated_df


def data_scale(X_DT):
    from sklearn.preprocessing import StandardScaler
    X_DT = X_DT.select_dtypes(include='number')
    cols = X_DT.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_DT)
    X_scaled = pd.DataFrame(X_scaled, columns=cols)

    return X_scaled

def split_data(X_DT, y_DT, size):
    from sklearn.model_selection import train_test_split
    return train_test_split(X_DT, y_DT, test_size=size, random_state=42)

def evaluate_classifier(y_true, y_pred):
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, precision_score

    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))
    print("===================")
    print()
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    print("===================")

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)

    print(f"accuracy: {acc:.3f}")
    print(f"recall: {rec:.3f}")
    print(f"f1-score: {f1:.3f}")
    print(f"precision: {prec:.3f}")