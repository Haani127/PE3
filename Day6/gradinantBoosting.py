from sklearn.ensemble import GradientBoostingClassifier

X_train = [
    [1, 3],
    [2, 4],
    [2, 2],
    [3, 5],
    [4, 6],
    [5, 7],
    [6, 8],
    [7, 9]
]

y_train = [0, 0, 0, 0, 1, 1, 1, 1]

model = GradientBoostingClassifier(
    n_estimators=100,     # number of trees
    learning_rate=0.1,    # how strongly each tree corrects errors
    max_depth=2           # depth of each tree
)

model.fit(X_train, y_train)

test_data = [(2, 4), (6, 9)]

predictions = model.predict(test_data)

for data, result in zip(test_data, predictions):
    status = "Pass" if result == 1 else "Fail"
    print(f"Study Time: {data[0]}, Sleep Time: {data[1]} → {status}")
