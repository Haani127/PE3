import pandas as pd
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

file = input().strip()
filename = os.path.join(sys.path[0], file)

if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found.")
    sys.exit()

df = pd.read_csv(filename)

dept_dummies = pd.get_dummies(df['Department'], prefix='Department')
salary_dummies = pd.get_dummies(df['salary'], prefix='salary')

df = pd.concat([df, dept_dummies, salary_dummies], axis=1)

df.drop(['Department', 'salary'], axis=1, inplace=True)

X = df.drop('left', axis=1)
y = df['left']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3
)

model = GaussianNB()
model.fit(X_train, y_train)

print(model)
print()

predictions = model.predict(X_test)

print(predictions)
