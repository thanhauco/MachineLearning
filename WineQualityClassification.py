# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load dataset from UCI repository
red_wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
white_wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

# Read the datasets
red_wine = pd.read_csv(red_wine_url, sep=';')
white_wine = pd.read_csv(white_wine_url, sep=';')

# Combine red and white wine datasets into one dataframe
red_wine['wine_type'] = 'red'
white_wine['wine_type'] = 'white'
wine_data = pd.concat([red_wine, white_wine], axis=0)

# Preview the dataset
print(wine_data.head())

# Convert 'wine_type' into binary values (red = 0, white = 1)
wine_data['wine_type'] = wine_data['wine_type'].apply(lambda x: 1 if x == 'white' else 0)

# Define features (X) and target (y)
X = wine_data.drop(columns=['quality'])
y = wine_data['quality']  # Quality as the target (classification)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preview the split data
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Define and train AdaBoost model
ada_clf = AdaBoostClassifier(n_estimators=200, random_state=42)
ada_clf.fit(X_train, y_train)
ada_pred = ada_clf.predict(X_test)
ada_accuracy = accuracy_score(y_test, ada_pred)
print(f"AdaBoost Accuracy: {ada_accuracy * 100:.2f}%")

# Define and train Gradient Boosting model
gb_clf = GradientBoostingClassifier(n_estimators=200, random_state=42)
gb_clf.fit(X_train, y_train)
gb_pred = gb_clf.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
print(f"Gradient Boosting Accuracy: {gb_accuracy * 100:.2f}%")

# Define and train XGBoost model
xgb_clf = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_clf.fit(X_train, y_train)
xgb_pred = xgb_clf.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Accuracy: {xgb_accuracy * 100:.2f}%")

# Comparing all models
print("\nComparison of the three boosting algorithms on Wine Quality dataset:")
print(f"AdaBoost Accuracy: {ada_accuracy * 100:.2f}%")
print(f"Gradient Boosting Accuracy: {gb_accuracy * 100:.2f}%")
print(f"XGBoost Accuracy: {xgb_accuracy * 100:.2f}%")