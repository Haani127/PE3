import pandas as pd
import sys, os
import warnings
warnings.filterwarnings("ignore")

from ML_Modules import model_stack_classifier, evaluate_multilabel_classifier


filename = input().strip()

df = pd.read_csv(os.path.join(sys.path[0],filename))
# Feature & Target Separation
X = df.drop('Diabetic', axis=1).values
y = df['Diabetic'].values

# Train & Predict
y_true, y_pred = model_stack_classifier(X, y)

# Evaluation
evaluate_multilabel_classifier(y_true, y_pred)

#-----------ML_MODULES.PY CODE BELOW FOR REFERENCE----------------

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    recall_score,
    f1_score,
    precision_score
)


# -------------------------------------------------
# Build Stacking Classifier
# -------------------------------------------------
def get_stacking_cls():
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

    return model


# -------------------------------------------------
# Train with Repeated K-Fold CV
# -------------------------------------------------
def model_stack_classifier(X_cls, y_cls):
    rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42
    )

    model = get_stacking_cls()

    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in rkf.split(X_cls):
        X_train, X_test = X_cls[train_idx], X_cls[test_idx]
        y_train, y_test = y_cls[train_idx], y_cls[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

    y_true_st = np.concatenate(y_true_all)
    y_pred_st = np.concatenate(y_pred_all)

    acc = accuracy_score(y_true_st, y_pred_st)
    print(f"Accuracy: {acc:.3f}\n")

    return y_true_st, y_pred_st


# -------------------------------------------------
# Evaluation Metrics
# -------------------------------------------------
def evaluate_multilabel_classifier(y_true, y_pred):
    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))
    print("===================\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=2))
    print("===================")

    print(f"accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"recall: {recall_score(y_true, y_pred, average='weighted'):.3f}")
    print(f"f1-score: {f1_score(y_true, y_pred, average='weighted'):.3f}")
    print(f"precision: {precision_score(y_true, y_pred, average='weighted'):.3f}")