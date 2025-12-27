import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load Excel file
df = pd.read_excel(
    r"D:\PE3\portal-questions\Day3\ML470_S3_Diabetes_Data_Preprocessed_Concept.xlsx"
)

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)
model.fit(X_train, y_train)

# Visualize Decision Tree
plt.figure(figsize=(22, 12))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Non-Diabetic", "Diabetic"],
    filled=True,
    rounded=True,
    impurity=True,
    fontsize=10
)
plt.title("Decision Tree Visualization for Diabetes Prediction")
plt.show()
