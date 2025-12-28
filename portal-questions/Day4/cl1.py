#-----------------main--------------

import pandas as pd
import numpy as np
import warnings
import os
import sys
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score
from ML_Modules import split_data

filename = input().strip()

try:
    df = pd.read_csv(os.path.join(sys.path[0] , filename))
except:
    print(f"Error: File '{filename}' not found.")
    exit()

# Data Preview
print(df.head())
print()

X = df[['Fasting blood', 'bmi', 'age', 'FamilyHistory', 'HbA1c']].values
y = df[['target']].values

# ---------------- K-FOLD ----------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_scores = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    kf_scores.append(round(acc, 3))

kf_scores = np.array(kf_scores)
print("K-Fold Accuracy Scores:", kf_scores)
print("Mean CV Accuracy:", format(np.mean(kf_scores), ".3f"))

# ---------------- HOLD OUT ----------------
X_train, X_test, y_train, y_test = split_data(X, y, 0.3)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)
hold_acc = accuracy_score(y_test, preds)

print("Hold-Out Method Accuracy:", format(hold_acc, ".3f"))

# ---------------- LOOCV ----------------
loo = LeaveOneOut()
loo_preds = []
loo_actual = []

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    loo_preds.append(pred[0])
    loo_actual.append(y_test[0][0])

loo_acc = accuracy_score(loo_actual, loo_preds)
print("LOOCV Accuracy:", format(loo_acc, ".3f"))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_scores = []

for train_idx, test_idx in skf.split(X, y.ravel()):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    skf_scores.append(accuracy_score(y_test, preds))

print("Accuracy:", format(np.mean(skf_scores), ".3f"))

#---------------ML_Model-----------

from sklearn.model_selection import train_test_split

def split_data(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, random_state=42)