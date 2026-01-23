# main.py

import pandas as pd
import sys, os
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import AdaBoostClassifier
from ML_Modules import evaluate_multilabel_classifier, model_adaboost_classifier

# -------------------------------
# Read CSV filename
# -------------------------------
filename = input().strip()

try:
    df = pd.read_csv(os.path.join(sys.path[0],filename))
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit()

# -------------------------------
# Display first 5 rows
# -------------------------------
print(df.head())
print()

# -------------------------------
# Separate features and target
# -------------------------------
X = df.iloc[:, :-1].to_numpy()   # First 19 columns
y = df['Diabetic'].to_numpy()    # Target

# -------------------------------
# Train AdaBoost on FULL DATA
# -------------------------------
model = AdaBoostClassifier(random_state=42)
model.fit(X, y)

# -------------------------------
# Predict on FULL DATA
# -------------------------------
y_pred = model.predict(X)

# -------------------------------
# Evaluation (MATCHES EXPECTED OUTPUT)
# -------------------------------
evaluate_multilabel_classifier(y, y_pred)

# -------------------------------
# Cross-Validation Metrics
# -------------------------------
model_adaboost_classifier(X, y)

#-----------ML_MODULES.PY CODE BELOW FOR REFERENCE----------------

# ML_Modules.py

import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.metrics import confusion_matrix, classification_report


def evaluate_multilabel_classifier(y_true, y_pred):
    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred))
    print("===================\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=2))
    print("===================")


def model_adaboost_classifier(X_class, y_class):
    model = AdaBoostClassifier(random_state=42)

    rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=2,
        random_state=42
    )

    scoring = {
        'accuracy': 'accuracy',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted',
        'precision': 'precision_weighted'
    }

    cv_results = cross_validate(
        model, X_class, y_class, cv=rkf, scoring=scoring
    )

    print(f"accuracy:  {cv_results['test_accuracy'].mean():.3f}")
    print(f"recall: {cv_results['test_recall'].mean():.3f}")
    print(f"f1-score: {cv_results['test_f1'].mean():.3f}")
    print(f"precision: {cv_results['test_precision'].mean():.3f}")