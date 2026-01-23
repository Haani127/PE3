import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import sys,os

# Read CSV file name
file_name = input().strip()
file = os.path.join(sys.path[0],file_name)

# Load dataset
df = pd.read_csv(file)

# Convert quality to binary target
df['target'] = (df['quality'] >= 7).astype(int)

# Features and target
X = df.drop(['quality', 'target'], axis=1)
y = df['target']

# Models
models = [
    ("Naive Bayes", GaussianNB()),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
    ("SVM", SVC(kernel='linear', probability=True, random_state=42))
]

# 5-Fold Cross-Validation
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for name, model in models:
    acc_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    prec_scores = cross_val_score(model, X, y, cv=kf, scoring=make_scorer(precision_score, average='binary',zero_division=0))
    rec_scores = cross_val_score(model, X, y, cv=kf, scoring=make_scorer(recall_score, average='binary',zero_division=0))
    f1_scores = cross_val_score(model, X, y, cv=kf, scoring=make_scorer(f1_score, average='binary',zero_division=0))

    print(f"{name} Cross-Validation Results:")
    print(f"Average Accuracy: {np.mean(acc_scores):.2f}")
    print(f"Average Precision: {np.mean(prec_scores):.2f}")
    print(f"Average Recall: {np.mean(rec_scores):.2f}")
    print(f"Average F1-score: {np.mean(f1_scores):.2f}\n")
