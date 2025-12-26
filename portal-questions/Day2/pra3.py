import pandas as pd
import os
import sys
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import ML_Modules as mm

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")

# Read filename
file = input().strip()
filename=os.path.join(sys.path[0],file)
# Check file existence
if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found.")
    sys.exit()

# Load dataset
df = pd.read_csv(filename)

# Label Encoding
salary_encoder = LabelEncoder()
dept_encoder = LabelEncoder()

df['salary.enc'] = salary_encoder.fit_transform(df['salary'])
df['Department.enc'] = dept_encoder.fit_transform(df['Department'])

print("=== Label Encoding Categorical Columns ===")
print(f"Encoded salary classes: {list(salary_encoder.classes_)}")
print(f"Encoded Department classes: {list(dept_encoder.classes_)}")

# Drop original categorical columns
df = df.drop(columns=['salary', 'Department'])

# Separate features and label
print("\n=== Separating Features and Label ===")
X = df.drop(columns=['left'])
y = df['left']

print(f"Input Features Shape: {X.shape}")
print(f"Label Shape: {y.shape}")

# Correlation matrix
print("\n=== Correlation Boolean Matrix (correlation >= 0.75) ===")
corr_bool = mm.check_correlation(X)
print(corr_bool)

# Scale features
scaled_X = mm.data_scale(X)

print("\n=== Scaled Feature Sample (First 5 Rows) ===")
print(scaled_X.head())

# Train-test split
print("\n=== Splitting Data into Train (80%) and Test (20%) ===")
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=42
)

print(f"Training Features Shape: {X_train.shape}")
print(f"Training Labels Shape: {y_train.shape}")
print(f"Testing Features Shape: {X_test.shape}")
print(f"Testing Labels Shape: {y_test.shape}")

#------------- ML_Model------------------

import pandas as pd
from sklearn.preprocessing import StandardScaler

def check_correlation(input_df):
    corr_matrix = input_df.corr().abs()
    return corr_matrix >= 0.75

def data_scale(input_df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(input_df)
    return pd.DataFrame(scaled, columns=input_df.columns)