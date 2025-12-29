import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Create dataset
# -----------------------------
np.random.seed(42)

data = {
    "Age": np.random.randint(18, 60, 20),
    "Income": np.random.randint(20000, 100000, 20),
    "Visits": np.random.randint(1, 15, 20),
    "Purchase": np.random.randint(0, 2, 20)
}

df = pd.DataFrame(data)

# -----------------------------
# Bootstrap sample (same size)
# -----------------------------
bootstrap_df = df.sample(n=len(df), replace=True, random_state=1)

# -----------------------------
# Train Decision Tree on bootstrap
# -----------------------------
X_boot = bootstrap_df.drop("Purchase", axis=1)
y_boot = bootstrap_df["Purchase"]

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_boot, y_boot)

# -----------------------------
# Random Forest (many bootstraps)
# -----------------------------
X = df.drop("Purchase", axis=1)
y = df["Purchase"]

rf = RandomForestClassifier(
    n_estimators=5,
    bootstrap=True,
    random_state=42
)
rf.fit(X, y)

print("Original Dataset:\n", df)
print("\nBootstrap Sample:\n", bootstrap_df)
print("\nRandom Forest Trained with Bootstrapping")
