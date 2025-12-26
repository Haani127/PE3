from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# X = [Time of Day, Tired]
X = [
    [0, 1],  # Morning, Tired → Drink Coffee
    [0, 0],  # Morning, Not Tired → No
    [1, 1],  # Afternoon, Tired → Drink Coffee
    [1, 0]   # Afternoon, Not Tired → No
]

# y = Drink Coffee
y = [1, 0, 1, 0]

# Create decision tree
model = DecisionTreeClassifier(criterion="gini")

# Train the model
model.fit(X, y)

# Visualize the tree
plt.figure(figsize=(7,4))
plot_tree(
    model,
    feature_names=["Time of Day", "Tired"],
    class_names=["No Coffee", "Drink Coffee"],
    filled=True
)
plt.show()
