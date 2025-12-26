import pandas as pd
import sys
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import ML_Modules

warnings.simplefilter(action='ignore')

# Read filename
file = input().strip()
filename=os.path.join(sys.path[0],file)
# Check file existence
if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found.")
    sys.exit()

# Load dataset
df = pd.read_csv(filename)

# Encode categorical variables
salary_dummies = pd.get_dummies(df['salary'], prefix='salary')
dept_dummies = pd.get_dummies(df['Department'], prefix='dept')

df_encoded = pd.concat([df, salary_dummies, dept_dummies], axis=1)

# Drop original categorical columns
df_encoded = df_encoded.drop(columns=['salary', 'Department'])

# Split features and target
X = df_encoded.drop(columns='left')
y = df_encoded['left']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

# Train Gaussian Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
ML_Modules.evaluate_classifier(y_test, y_pred)

# AUC-ROC availability message
print("code is available inside the 'ML_Modules.py' file")


#----------------------------ML_Modules.py------------------------------------------------

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

def evaluate_classifier(y_test, y_pred):
    # Confusion Matrix
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("===================")

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=2))
    print("===================")

    # Individual metrics (weighted averages)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']

    print(f"accuracy: {acc:.3f}")
    print(f"recall: {recall:.3f}")
    print(f"f1-score: {f1:.3f}")
    print(f"precision: {precision:.3f}")


def auc_roc(classifier, X_test, y_test):
    y_prob = classifier.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_prob)
    return score