import pandas as pd
import sys
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
file = input().strip()
filename=os.path.join(sys.path[0],file)

if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found.")
    sys.exit()

# Load dataset
df = pd.read_csv(filename)

# Features and target
X = df['HealthText']
y = df['Outcome']

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# New sample prediction
new_sample = ["Age group: Senior | BMI status: Overweight | Glucose category: Very High Glucose Level"]
new_sample_vec = vectorizer.transform(new_sample)
prediction = model.predict(new_sample_vec)

# Output
print(f"Prediction: {prediction[0]}")