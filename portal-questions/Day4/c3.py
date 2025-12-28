import os,sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from ML_Modules import split_data

filename = input().strip()

try:
    df = pd.read_csv(os.path.join(sys.path[0],filename))
except:
    print(f"Error: File '{filename}' not found.")
    exit()

# Data Preview
print(df.head())
print()

X = df[['bmi', 'age', 'insulin', 'FamilyHistory', 'bp']].values
y = df[['Fasting blood']].values

# Train-Test Split
X_train, X_test, y_train, y_test = split_data(X, y, 0.3)

# Hyperparameter Grid
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

model = DecisionTreeRegressor(random_state=42)

grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid.fit(X_train, y_train.ravel())

best_params = dict(sorted(grid.best_params_.items()))
print("Best Hyperparameters:", best_params)

best_model = grid.best_estimator_

# Cross-Validation RMSE on Full Dataset
cv_scores = cross_val_score(
    best_model,
    X,
    y.ravel(),
    cv=5,
    scoring='neg_mean_squared_error'
)

rmse_scores = np.sqrt(-cv_scores)
print("Cross-Validation RMSE Scores:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())

# Test RMSE
best_model.fit(X_train, y_train)
test_preds = best_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
print("RMSE:", test_rmse)

# Std Deviation of Label
std_dev = df['Fasting blood'].std()
print("Standard Deviation of Label:", std_dev)

# Interpretation
if test_rmse <= std_dev:
    print("The model's RMSE is within the standard deviation, indicating good performance.")
else:
    print("The model's RMSE exceeds the standard deviation, suggesting room for improvement.")

#----------------------------ML MODULES: split_data FUNCTION----------------------------


from sklearn.model_selection import train_test_split

def split_data(X, y, test_size):
    return train_test_split(X, y, test_size=test_size, random_state=42)