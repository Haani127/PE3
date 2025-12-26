import pandas as pd
import sys
import os

# Read filename
file = input().strip()
filename=os.path.join(sys.path[0],file)
# Check file existence
if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found.")
    sys.exit()

# Load dataset
df = pd.read_csv(filename)

# 1. First 5 rows
print("First 5 rows of the dataset:")
print(df.head())
print()

# 2. Number of samples
print("Number of samples in the data:")
print(df.shape[0])
print()

# 3. Data types
print("Data types of each column:")
print(df.dtypes)
print()

# 4. Feature columns
features = [
    'satisfaction_level',
    'last_evaluation',
    'number_project',
    'average_montly_hours',
    'time_spend_company',
    'Work_accident',
    'promotion_last_5years'
]
print("Feature columns used for classification:")
print(features)
print()

# 5. Statistical summary
print("Statistical summary of numeric columns:")
print(df.describe())
print()

# 6. Sample categorical data
if all(col in df.columns for col in ['Department', 'salary', 'left']):
    print("Sample categorical data (Department, salary, left):")
    print(df[['Department', 'salary', 'left']].head())
else:
    print("Categorical columns ('Department', 'salary', 'left') not found in dataset.")