import pandas as pd
import sys,os
import warnings
warnings.filterwarnings("ignore")

from ML_Modules import (
    model_adaboost_regressor,
    model_stacking_regressor
)

filename = input().strip()

try:
    df = pd.read_csv(os.path.join(sys.path[0], filename))
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

if "price" not in df.columns:
    print("Error: price column not found.")
    sys.exit(1)

X = df.drop("price", axis=1).values
y = df["price"].values

print("=== AdaBoost Regressor Performance ===")
model_adaboost_regressor(X, y)

print("\n=== Stacking Regressor Performance ===")
model_stacking_regressor(X, y)

#-----------ML_MODULES.PY CODE BELOW FOR REFERENCE----------------

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import AdaBoostRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score


def evaluate_regressor(y_true, y_pred):
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"R-squared: {r2:.3f}")


def model_adaboost_regressor(X_reg, y_reg):

    rkf = RepeatedKFold(
        n_splits=10,
        n_repeats=3,
        random_state=42
    )

    model = AdaBoostRegressor(
        base_estimator=DecisionTreeRegressor(random_state=42),
        n_estimators=50,
        learning_rate=1.0,
        random_state=42
    )

    y_true_ab = []
    y_pred_ab = []

    for train_idx, test_idx in rkf.split(X_reg):
        X_train, X_test = X_reg[train_idx], X_reg[test_idx]
        y_train, y_test = y_reg[train_idx], y_reg[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_ab.append(y_test)
        y_pred_ab.append(y_pred)

    evaluate_regressor(y_true_ab, y_pred_ab)


def get_stacking():

    level0 = [
        ('lr', LinearRegression()),
        ('knn', KNeighborsRegressor()),
        ('dt', DecisionTreeRegressor(random_state=42))
    ]

    meta_estimator = LinearRegression()

    model = StackingRegressor(
        estimators=level0,
        final_estimator=meta_estimator,
        cv=5
    )

    return model


def model_stacking_regressor(X_reg, y_reg):

    rkf = RepeatedKFold(
        n_splits=10,
        n_repeats=3,
        random_state=42
    )

    model = get_stacking()

    y_true_st = []
    y_pred_st = []

    for train_idx, test_idx in rkf.split(X_reg):
        X_train, X_test = X_reg[train_idx], X_reg[test_idx]
        y_train, y_test = y_reg[train_idx], y_reg[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_true_st.append(y_test)
        y_pred_st.append(y_pred)

    evaluate_regressor(y_true_st, y_pred_st)