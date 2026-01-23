import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

# Function to load dataset and train the model
def predict_spam():
    # Prompt user for dataset filename
    file_name = input()
    
    # Construct the file path
    file_path = os.path.join(sys.path[0], file_name)
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {file_name} not found.")
        return

    # Load the dataset
    df = pd.read_csv(file_path)

    # Handle missing values (fill with mean for numerical columns)
    if df.isnull().sum().any():
        print("Missing values detected. Handling missing data by filling with mean for numerical columns.")
        df.fillna(df.mean(), inplace=True)

    # Features and target variable
    X = df['Email Content']
    y = df['Label'].apply(lambda x: 1 if x == 'Spam' else 0)  # Convert target to binary (1 for Spam, 0 for Not Spam)

    # Vectorize the email content using CountVectorizer
    vectorizer = CountVectorizer(stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train the Gaussian Naive Bayes model
    model = GaussianNB()
    model.fit(X_train.toarray(), y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test.toarray())

    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print the results in the required output format
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))

# Run the prediction function
predict_spam()
