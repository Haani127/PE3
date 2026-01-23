import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os,sys
# Read CSV file
file_name = input().strip()
file = os.path.join(sys.path[0],file_name)
df = pd.read_csv(file)

# Convert quality to binary target
df['target'] = (df['quality'] >= 7).astype(int)

# Features and target
X = df.drop(['quality', 'target'], axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# --- AdaBoost ---
base_est = DecisionTreeClassifier(max_depth=1, random_state=42)
ada_model = AdaBoostClassifier(base_estimator=base_est, n_estimators=50, random_state=42)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)

acc_ada = accuracy_score(y_test, y_pred_ada)
prec_ada = precision_score(y_test, y_pred_ada, zero_division=0)
rec_ada = recall_score(y_test, y_pred_ada, zero_division=0)
f1_ada = f1_score(y_test, y_pred_ada, zero_division=0)
cm_ada = confusion_matrix(y_test, y_pred_ada)

print("AdaBoost Classifier Results:")
print(f"Accuracy: {acc_ada:.2f}")
print(f"Precision: {prec_ada:.2f}")
print(f"Recall: {rec_ada:.2f}")
print(f"F1-score: {f1_ada:.2f}")
print("Confusion Matrix:")
print(cm_ada)
print()

# --- XGBoost ---

