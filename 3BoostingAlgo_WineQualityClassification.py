# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load Wine Quality Dataset (Red and White Wine)
red_wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
white_wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

# Read the datasets
red_wine = pd.read_csv(red_wine_url, sep=';')
white_wine = pd.read_csv(white_wine_url, sep=';')

# Add a 'wine_type' column to differentiate between red (0) and white (1) wine
red_wine['wine_type'] = 0  # Red wine = 0
white_wine['wine_type'] = 1  # White wine = 1

# Combine the two datasets
wine_data = pd.concat([red_wine, white_wine], axis=0)

# Define features (X) and target (y)
X = wine_data.drop(columns=['quality'])  # All columns except 'quality' are features
y = wine_data['quality']  # 'quality' is the target variable

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shape of the training and test datasets
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# AdaBoost Model
ada_clf = AdaBoostClassifier(n_estimators=200, random_state=42)
ada_clf.fit(X_train, y_train)
ada_pred = ada_clf.predict(X_test)
ada_accuracy = accuracy_score(y_test, ada_pred)
print(f"AdaBoost Accuracy: {ada_accuracy * 100:.2f}%")

# Gradient Boosting Model
gb_clf = GradientBoostingClassifier(n_estimators=200, random_state=42)
gb_clf.fit(X_train, y_train)
gb_pred = gb_clf.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
print(f"Gradient Boosting Accuracy: {gb_accuracy * 100:.2f}%")

# XGBoost Model
xgb_clf = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_clf.fit(X_train, y_train)
xgb_pred = xgb_clf.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Accuracy: {xgb_accuracy * 100:.2f}%")

# Comparison of all models
print("\nComparison of the three boosting algorithms on the Wine Quality dataset:")
print(f"AdaBoost Accuracy: {ada_accuracy * 100:.2f}%")
print(f"Gradient Boosting Accuracy: {gb_accuracy * 100:.2f}%")
print(f"XGBoost Accuracy: {xgb_accuracy * 100:.2f}%")

# Hyperparameter Tuning with Grid Search for AdaBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 1.0]
}

grid_search = GridSearchCV(AdaBoostClassifier(random_state=42), param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Print the best parameters and accuracy for AdaBoost after hyperparameter tuning
print(f"\nBest parameters for AdaBoost: {grid_search.best_params_}")
print(f"Best AdaBoost accuracy with tuned parameters: {grid_search.best_score_ * 100:.2f}%")

# Apply the tuned AdaBoost model to the test data
best_ada_clf = grid_search.best_estimator_
best_ada_pred = best_ada_clf.predict(X_test)
best_ada_accuracy = accuracy_score(y_test, best_ada_pred)
print(f"Test accuracy with tuned AdaBoost: {best_ada_accuracy * 100:.2f}%")