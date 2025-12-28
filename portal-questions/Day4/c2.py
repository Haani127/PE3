import os,sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

filename = input().strip()

try:
    df = pd.read_csv(os.path.join(sys.path[0],filename))
except:
    print(f"Error: File '{filename}' not found.")
    exit()

X = df[['Fasting blood', 'bmi', 'age', 'FamilyHistory', 'HbA1c']].values
y = df[['target']].values.ravel()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {
    'max_depth': [2, 3, 4, 5, 6],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3]
}

model = DecisionTreeClassifier(random_state=42)

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X, y)

best_params = grid.best_params_
best_score = grid.best_score_

best_params = dict(sorted(best_params.items()))

print("Best Hyperparameters:", best_params)
print("Best Stratified CV Accuracy:", format(best_score, ".3f"))