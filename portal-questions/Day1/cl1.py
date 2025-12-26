import pandas as pd
import sys
import os
import ML_Modules

# Read filename
file = input().strip()
filename=os.path.join(sys.path[0],file)
# Check file existence
if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found.")
    sys.exit()

# Load CSV (dataset not used for calculation)
pd.read_csv(filename)

# Common probabilities
P_disease = 0.001
P_positive_given_disease = 0.95
P_no_disease = 1 - P_disease

# Scenario 1
print("Scenario 1: False positive rate = 5%")
ML_Modules.calc_probability(
    P_disease,
    P_positive_given_disease,
    0.05,
    P_no_disease
)
print()

# Scenario 2
print("Scenario 2: False positive rate = 10%")
ML_Modules.calc_probability(
    P_disease,
    P_positive_given_disease,
    0.10,
    P_no_disease
)

#----------------------------ML_Modules.py------------------------------------------------

def calc_probability(P_disease, P_positive_given_disease,
                     P_positive_given_no_disease, P_no_disease):

    numerator = P_positive_given_disease * P_disease
    denominator = numerator + (P_positive_given_no_disease * P_no_disease)

    if denominator == 0:
        print("Invalid probability values.")
        return None

    probability = numerator / denominator
    print(f"Probability of having disease given a positive test: {probability:.4f}")
    return probability