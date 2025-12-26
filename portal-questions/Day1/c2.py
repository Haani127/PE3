import pandas as pd
import numpy as np
import os
import sys
import warnings

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import ML_Modules

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

file = input().strip()
filename = os.path.join(sys.path[0], file)

if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found.")
    sys.exit()

df = pd.read_csv(filename)

print(df.head())
print()

print(df.dtypes)
print()

X = df[['Glucose', 'BMI', 'Age', 'FamilyHistory', 'HbA1c']]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

model = GaussianNB()
model.fit(X_train, y_train)

print("Model trained.")
print()

y_pred = model.predict(X_test)
print(f"Predicted Values: array({list(np.array(y_pred))})")
print()

ML_Modules.evaluate_classifier(y_test, y_pred)


#------ML_Modules.py----------------------------

from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score, precision_score

def evaluate_classifier(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    print("Confusion Matrix")
    print(cm)
    print("===================")
    print(f"accuracy: {acc:.3f}")
    print(f"recall: {recall:.3f}")
    print(f"f1-score: {f1:.3f}")
    print(f"precision: {precision:.3f}")
