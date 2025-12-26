import pandas as pd
import sys
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import ML_Modules as mm
import os
warnings.simplefilter(action='ignore')

# Read input filename
file= input().strip()
filename=os.path.join(sys.path[0],file)
# File check
try:
    df = pd.read_csv(filename)
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit()

# Check target column
if 'left' not in df.columns:
    print("Error: 'left' column not found.")
    sys.exit()

# Label Encoding
salary_le = LabelEncoder()
dept_le = LabelEncoder()

df['salary.enc'] = salary_le.fit_transform(df['salary'])
df['Department.enc'] = dept_le.fit_transform(df['Department'])

# Drop original categorical columns
df.drop(columns=['salary', 'Department'], inplace=True)

# Separate features and label
X = df.drop(columns=['left'])
y = df['left']

# Scale features
X_scaled = mm.data_scale(X)

# Sequential split (80-20)
X_train, X_test, y_train, y_test = mm.sequential_split(X_scaled, y, 0.8)

# Train SVM model
model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mm.evaluate_classifier(y_test, y_pred)


#---------ML_Model------------------

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
    scaled_data = scaler.fit_transform(df)
    return pd.DataFrame(scaled_data, columns=df.columns)

def sequential_split(X, y, train_ratio=0.8):
    split_index = int(len(X) * train_ratio)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    return X_train, X_test, y_train, y_test

def evaluate_classifier(y_test, y_pred):
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("===================\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))
    print("===================\n")

    print(f"accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"recall: {recall_score(y_test, y_pred):.3f}")
    print(f"f1-score: {f1_score(y_test, y_pred):.3f}")
    print(f"precision: {precision_score(y_test, y_pred):.3f}")