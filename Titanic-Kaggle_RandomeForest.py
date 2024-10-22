# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Load the dataset
try:
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(url)
except Exception as e:
    print(f"Error loading the dataset: {e}")
    print("Please check your internet connection or use a local file path.")
    exit()

# Data preview
print(df.head())
print("\nDataset Info:")
print(df.info())

# Data Preprocessing
# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Fill missing Age values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked values with the mode (most frequent value)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Fill missing Fare values with the median
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Drop columns that won't be used for prediction
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Convert categorical variables to dummy/indicator variables (one-hot encoding)
categorical_columns = ['Sex', 'Embarked']
df = pd.get_dummies(df, columns=[col for col in categorical_columns if col in df.columns])

# Ensure 'Survived' column exists
if 'Survived' not in df.columns:
    print("Error: 'Survived' column not found in the dataset.")
    exit()

# Define features (X) and target (y)
X = df.drop(columns='Survived')
y = df['Survived']

# Split data into training and testing sets
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except Exception as e:
    print(f"Error splitting the data: {e}")
    exit()

# Initialize and train a Random Forest Classifier
try:
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
except Exception as e:
    print(f"Error training the model: {e}")
    exit()

# Make predictions on the test set
try:
    y_pred = clf.predict(X_test)
except Exception as e:
    print(f"Error making predictions: {e}")
    exit()

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Output feature importance
importances = clf.feature_importances_
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importance = feature_importance.sort_values(by='importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)