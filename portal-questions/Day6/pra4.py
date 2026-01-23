import pandas as pd
import sys, os
import warnings
warnings.filterwarnings("ignore")

from ML_Modules import data_scale, model_stacking_classifier


filename = input().strip()

try:
    df = pd.read_csv(os.path.join(sys.path[0], filename))
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

X_df = df.loc[:, df.columns != "price_range"]
y = df["price_range"].values

X_scaled = data_scale(X_df)

X = X_scaled.values

model_stacking_classifier(X, y)

#-----------ML_MODULES.PY CODE BELOW FOR REFERENCE----------------

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    recall_score,
    f1_score,
    precision_score
)


def data_scale(X_DT):
    scaler = StandardScaler()
    numeric_cols = X_DT.select_dtypes(include='number').columns
    scaled = scaler.fit_transform(X_DT[numeric_cols])
    return pd.DataFrame(scaled, columns=numeric_cols)


def evaluate_multilabel_classifier(y_true, y_pred):

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))
    print("===================\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=2))
    print("===================\n")

    print(f"accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"recall: {recall_score(y_true, y_pred, average='weighted'):.3f}")
    print(f"f1-score: {f1_score(y_true, y_pred, average='weighted'):.3f}")
    print(f"precision: {precision_score(y_true, y_pred, average='weighted'):.3f}")


def model_stacking_classifier(X, y):

    rskf = RepeatedStratifiedKFold(
        n_splits=10,
        n_repeats=3,
        random_state=42
    )

    estimators = [
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('knn', KNeighborsClassifier()),
        ('dt', DecisionTreeClassifier(random_state=42))
    ]

    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5
    )

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

    acc = accuracy_score(np.concatenate(y_true_all), np.concatenate(y_pred_all))
    print(f"Accuracy: {acc:.3f}\n")

    evaluate_multilabel_classifier(y_true_all, y_pred_all)
