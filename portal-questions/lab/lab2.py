import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import os,sys
# Read input filename
file_name = input().strip()
file = os.path.join(sys.path[0],file_name)
# Load dataset
df = pd.read_csv(file)

# Encode target labels
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

X = df.drop('species', axis=1)
y = df['species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Class labels in required order
class_labels = ['setosa', 'versicolor', 'virginica']

# SVM models with different kernels
models = [
    ("SVM (Linear Kernel)", SVC(kernel='linear', random_state=42)),
    ("SVM (RBF Kernel)", SVC(kernel='rbf', random_state=42))
]

# Train and evaluate models
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(name)
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision (macro): {prec:.2f}")
    print(f"Recall (macro): {rec:.2f}")
    print(f"F1-score (macro): {f1:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_labels,
            digits=3,
            zero_division=0
        )
    )
