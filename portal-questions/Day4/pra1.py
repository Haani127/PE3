# Data Preparation for Regression – Average Monthly Hours
import os,sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv(os.path.join(sys.path[0],input()))

print(f"The number of samples in data is {df.shape[0]}.")

# -----------------------------
# 1. Data Types
# -----------------------------
print("Data Types:")
print(df.dtypes)

# -----------------------------
# 2. Numeric Summary
# -----------------------------
print("\nNumeric Summary:")
print(df.describe())

# -----------------------------
# 3. Drop Irrelevant Columns
# -----------------------------
df_clean = df.drop(columns=["Department", "salary"])

print("\nData After Dropping Irrelevant Columns:")
print(df_clean.info())

# -----------------------------
# 4. Select Features and Target
# -----------------------------
X = df_clean.drop(columns=["average_monthly_hours"])
y = df_clean["average_monthly_hours"]

print("\nInput Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

# -----------------------------
# 5. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\nScaled Feature Data:")
print(X_scaled.head())
