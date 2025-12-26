import pandas as pd
import sys
import os
import warnings

warnings.simplefilter(action='ignore')

# Read filename
file= input().strip()
filename=os.path.join(sys.path[0],file)
# Check file existence
if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found.")
    sys.exit()

# Load dataset safely
try:
    df = pd.read_csv(filename)
    if df.empty:
        raise ValueError
except:
    print(f"Error: File '{filename}' is empty or invalid.")
    sys.exit()

# 1. Rows with missing values
print("Rows with missing values (if any):")
missing_rows = df[df.isnull().any(axis=1)]

if missing_rows.empty:
    print("No missing values found in the dataset.\n")
else:
    print(missing_rows)
    print()

# 2. Correlation matrix of numeric columns
print("Correlation matrix of numeric columns:")
numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()
print(corr_matrix)
print()