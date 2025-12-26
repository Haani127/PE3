import pandas as pd
import sys
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import ML_Modules as mm
import os
warnings.simplefilter(action='ignore')

# Read filename
file = input().strip()
filename=os.path.join(sys.path[0],file)
# Load dataset
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit()

# Encode categorical columns
salary_le = LabelEncoder()
dept_le = LabelEncoder()

df['salary.enc'] = salary_le.fit_transform(df['salary'])
df['Department.enc'] = dept_le.fit_transform(df['Department'])

# Drop original categorical columns
df.drop(columns=['salary', 'Department'], inplace=True)

# Feature-label split
X = df.drop(columns=['left'])
y = df['left']

# Scale features
X_scaled = mm.data_scale(X)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42
)

# Grid Search setup
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['rbf']
}

grid = GridSearchCV(
    estimator=SVC(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Train GridSearch
grid.fit(X_train, y_train)

# Best model prediction
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluation
mm.evaluate_classifier(y_test, y_pred)

#------------ML_Model---------------

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    recall_score,
    f1_score,
    precision_score
)

def data_scale(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    return pd.DataFrame(scaled, columns=df.columns)

def evaluate_classifier(y_test, y_pred):
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("===================")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("===================")

    print(f"accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"recall: {recall_score(y_test, y_pred):.3f}")
    print(f"f1-score: {f1_score(y_test, y_pred):.3f}")
    print(f"precision: {precision_score(y_test, y_pred):.3f}")