import pandas as pd
import sys
import os
import warnings
from sklearn.model_selection import train_test_split

warnings.simplefilter(action='ignore')

# Read filename
file= input().strip()
filename=os.path.join(sys.path[0],file)
# Check file existence
if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found.")
    sys.exit()

# Load dataset
df = pd.read_csv(filename)

# 1. Dummy variables for salary
print("Creating dummy variables for salary:")
salary_dummies = pd.get_dummies(df['salary'], prefix='salary')
df_salary = pd.concat([df, salary_dummies], axis=1)
print(df_salary.head())
print()

# 2. Dummy variables for department
print("Creating dummy variables for department:")
dept_dummies = pd.get_dummies(df_salary['Department'], prefix='dept')
df_final = pd.concat([df_salary, dept_dummies], axis=1)
print(df_final.head())
print()

# 3. Final dataframe
print("Final dataframe with dummy variables:")
print(df_final.head())
print()

# 4. Train-test split
train_df, test_df = train_test_split(df_final, train_size=0.7)

print("Size of training dataset: ", train_df.shape)
print("Size of test dataset: ", test_df.shape)
print()

# 5. Feature separation
X_train = train_df.drop(columns='left')
y_train = train_df['left']

X_test = test_df.drop(columns='left')
y_test = test_df['left']

print("Shapes of input/output features after train-test split:")
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)