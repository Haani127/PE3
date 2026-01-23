import pandas as pd
import sys, os
import warnings
warnings.filterwarnings("ignore")

from ML_Modules import model_stack_classifier


filename = input().strip()


df = pd.read_csv(os.path.join(sys.path[0], filename))

X = df.drop("price_range.enc", axis=1).values
y = df["price_range.enc"].values

model_stack_classifier(X, y)

#-----------ML_MODULES.PY CODE BELOW FOR REFERENCE----------------

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    recall_score,
    f1_score,
    precision_score
)


def model_stack_classifier(X_cls, y_cls):

    X_train, X_test, y_train, y_test = train_test_split(
        X_cls,
        y_cls,
        test_size=0.3,
        random_state=42,
        stratify=y_cls
    )

    estimators = [
        ('lr', LogisticRegression()),
        ('knn', KNeighborsClassifier()),
        ('dt', DecisionTreeClassifier(random_state=42))
    ]

    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}\n")


    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("===================\n")


    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=2))
    print("===================")

    print(f"accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"recall: {recall_score(y_test, y_pred, average='weighted'):.3f}")
    print(f"f1-score: {f1_score(y_test, y_pred, average='weighted'):.3f}")
    print(f"precision: {precision_score(y_test, y_pred, average='weighted'):.3f}")