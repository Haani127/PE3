import pandas as pd
import sys
import os
from sklearn.preprocessing import LabelEncoder

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
print("=== First 5 Rows of Data ===")
print(df.head())
print()

# 2. Number of samples
print(f"The number of samples in data is {df.shape[0]}.\n")

# 3. Data types
print("=== Data Types ===")
print(df.dtypes)
print()

# 4. Statistical summary
print("=== Statistical Summary (Describe) ===")
print(df.describe())
print()

# 5. Missing values
print("=== Missing Values Per Column ===")
print(df.isnull().sum())
print()

# 6. Salary encoding
salary_encoder = LabelEncoder()
df['salary.enc'] = salary_encoder.fit_transform(df['salary'])

print("=== Salary Encoding Classes ===")
print(list(salary_encoder.classes_))
print()

# 7. Department encoding
dept_encoder = LabelEncoder()
df['Department.enc'] = dept_encoder.fit_transform(df['Department'])

print("=== Department Encoding Classes ===")
print(list(dept_encoder.classes_))
print()

# 8. Drop original categorical columns
print("=== Dropping 'Department' and 'salary' columns ===")
df = df.drop(columns=['Department', 'salary'])

# 9. Updated DataFrame info
print("\n=== Updated DataFrame Info ===")
df.info()