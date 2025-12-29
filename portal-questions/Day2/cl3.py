import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score, f1_score

def evaluate_classifier(y_test, y_pred):
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("===================\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("===================")
    print(f"accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"recall: {recall_score(y_test, y_pred):.3f}")
    print(f"f1-score: {f1_score(y_test, y_pred):.3f}")
    print(f"precision: {precision_score(y_test, y_pred):.3f}")

def main():
    df = pd.read_csv(os.path.join(sys.path[0], input()))
    
    df = df[(df['Glucose'] != 0) & (df['BMI'] != 0)]
    features = ['Glucose', 'BMI', 'Age', 'FamilyHistory', 'HbA1c']
    X = df[features].copy()
    y = df['Outcome']
    
    for col in features:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        X[col] = X[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['rbf']
    }
    grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    y_pred = grid.best_estimator_.predict(X_test)
    evaluate_classifier(y_test, y_pred)

if __name__ == "__main__":
    main()


#----------------------------ML MODULES: data_scale FUNCTION----------------------------

import numpy as np
from sklearn.metrics import confusion_matrix

def iqr_winsorize(df, cols, threshold=1.5):
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    return df


def print_confusion(y_true, y_pred):
    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))
    print("===================")
    print()
