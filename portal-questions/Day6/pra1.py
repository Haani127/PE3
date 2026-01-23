import pandas as pd
import sys, os
import warnings
warnings.filterwarnings("ignore")

from ML_Modules import check_correlation_text, data_scale


filename = input().strip()

try:
    mobile_df = pd.read_csv(os.path.join(sys.path[0],filename))
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    sys.exit(1)

print("Dataset loaded successfully.\n")


print("Preview of dataset:")
print(mobile_df.head())

print("Dataset information:")
mobile_df.info()
print()

print("Missing values in each column:")
print(mobile_df.isnull().sum())
print()

X = mobile_df.loc[:, mobile_df.columns != "price_range"]
y = mobile_df["price_range"]
print("Input and target variables separated.\n")

print("Multicollinearity check:")
check_correlation_text(X)

scaled_X = data_scale(X)

print("Scaled input features (first 5 rows):")
print(scaled_X.head())

#-----------ML_MODULES.PY CODE BELOW FOR REFERENCE----------------

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler


def check_correlation_text(input_df):
    corr_matrix = abs(input_df.corr()) >= 0.7
    print(corr_matrix)


def data_scale(X_DT):
    scaler = StandardScaler()
    numeric_cols = X_DT.select_dtypes(include='number').columns
    scaled_data = scaler.fit_transform(X_DT[numeric_cols])
    return pd.DataFrame(scaled_data, columns=numeric_cols)
