import pandas as pd
import sys, os
import warnings
warnings.filterwarnings("ignore")

from ML_Modules import data_scale, model_adaboost_classifier


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

model_adaboost_classifier(X, y)
#-----------ML_MODULES.PY CODE BELOW FOR REFERENCE----------------

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_predict
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


def model_adaboost_classifier(X, y):

    rkf = RepeatedKFold(
        n_splits=10,
        n_repeats=1,
        random_state=42
    )

    model = AdaBoostClassifier(random_state=42)

    y_pred = cross_val_predict(model, X, y, cv=rkf)

    print("Confusion Matrix")
    print(confusion_matrix(y, y_pred))
    print("===================\n")

    print("Classification Report:")
    print(classification_report(y, y_pred, digits=2))
    print("===================")

    print(f"accuracy: {accuracy_score(y, y_pred):.3f}")
    print(f"recall: {recall_score(y, y_pred, average='weighted'):.3f}")
    print(f"f1-score: {f1_score(y, y_pred, average='weighted'):.3f}")
    print(f"precision: {precision_score(y, y_pred, average='weighted'):.3f}")
