import pandas as pd
import sys, os
import warnings
warnings.filterwarnings("ignore")

from ML_Modules import model_adaboost_classifier


filename = input().strip()

df = pd.read_csv(os.path.join(sys.path[0], filename))

print(df.head())

X = df.drop("price_range.enc", axis=1).values
y = df["price_range.enc"].values

model_adaboost_classifier(X, y)

#-----------ML_MODULES.PY CODE BELOW FOR REFERENCE----------------

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    recall_score,
    f1_score,
    precision_score
)

def model_adaboost_classifier(X_cls, y_cls):

    X_train, X_test, y_train, y_test = train_test_split(
        X_cls,
        y_cls,
        test_size=0.3,
        random_state=42,
        stratify=y_cls
    )

    base_estimator = DecisionTreeClassifier(
        max_depth=2,
        random_state=42
    )

    model = AdaBoostClassifier(
        base_estimator=base_estimator,   
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("===================")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=2))
    print("===================")

    print(f"accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"recall: {recall_score(y_test, y_pred, average='weighted'):.3f}")
    print(f"f1-score: {f1_score(y_test, y_pred, average='weighted'):.3f}")
    print(f"precision: {precision_score(y_test, y_pred, average='weighted'):.3f}")
