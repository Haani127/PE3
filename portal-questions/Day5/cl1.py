import pandas as pd
import numpy as np
import warnings
import sys
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")


filename = input().strip()

try:
    df = pd.read_csv(os.path.join(sys.path[0],filename))
except Exception:
    print(f"Error: Unable to read file '{filename}'.")
    sys.exit(1)


if 'target' not in df.columns:
    print("Error: Target column 'target' not found.")
    sys.exit(1)

X = df.iloc[:, :-1].values
y = df['target'].values


rskf = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=3,
    random_state=42
)


def create_model():
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        n_jobs=-1,
        oob_score=True,
        random_state=42
    )


all_true = []
all_pred = []

for train_idx, test_idx in rskf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = create_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    all_true.append(y_test)
    all_pred.append(preds)


all_true = np.concatenate(all_true)
all_pred = np.concatenate(all_pred)

accuracy = accuracy_score(all_true, all_pred)


final_model = create_model()
final_model.fit(X, y)

oob_score = final_model.oob_score_

print("=================================")
print(f"Accuracy: {accuracy:.3f}")
print(f"OOB Score: {oob_score:.3f}")
print("=================================")
