import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# ----------------------------
# User-created email dataset
# ----------------------------
data = {
    "text": [
        "Win a free lottery prize now",
        "Congratulations! You have won cash",
        "Limited offer, click the link today",
        "Meeting scheduled at 10 AM tomorrow",
        "Please find the attached report",
        "Are we still on for dinner tonight?",
        "Claim your free reward immediately",
        "Project deadline has been extended"
    ],
    "label": ["spam", "spam", "spam", "ham", "ham", "ham", "spam", "ham"]
}

df = pd.DataFrame(data)

X = df["text"]
y = df["label"]

# ----------------------------
# Models with same pipeline
# ----------------------------
models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000))
    ]),
    
    "KNN": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", KNeighborsClassifier(n_neighbors=3))
    ]),
    
    "SVM": Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LinearSVC())
    ])
}

# ----------------------------
# 5-Fold Cross Validation
# ----------------------------
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="f1_macro")
    print(name, "Mean F1-score:", scores.mean())
