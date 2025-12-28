import os,sys
import pandas as pd
import numpy as np
import warnings
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from ML_Modules import data_scale

warnings.simplefilter(action='ignore')

filename = input().strip()

try:
    df = pd.read_csv(os.path.join(sys.path[0],filename))
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    exit()

df = df.drop(columns=["Department", "salary"])

X = df.drop(columns=["average_monthly_hours"])
y = df["average_monthly_hours"]

# Scale features
X_scaled = data_scale(X)

# Initialize model
model = DecisionTreeRegressor(random_state=42)

# 5-fold cross-validation (negative MSE)
scores = cross_val_score(
    model,
    X_scaled,
    y,
    cv=5,
    scoring="neg_mean_squared_error"
)

mse = -scores.mean()
print(f"Cross-validated MSE: {mse}")

# Train final model
model.fit(X_scaled, y)

# Predict
predictions = model.predict(X_scaled)

# ---- IMPORTANT FORMATTING FIX ----
np.set_printoptions(
    formatter={'float_kind': lambda x: f"{x:.1f}".rstrip('0')}
)

print("Predictions:", predictions)


#----------------------------ML MODULES: data_scale FUNCTION----------------------------

import pandas as pd
from sklearn.preprocessing import StandardScaler

def data_scale(X_DT):
    numeric_cols = X_DT.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_DT[numeric_cols])
    return pd.DataFrame(X_scaled, columns=numeric_cols)
