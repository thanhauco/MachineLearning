import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate a sample dataset (replace this with your actual data loading)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Isolation Forest model
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)

# Fit the model
iso_forest.fit(X_train)

# Predict the anomalies (fraud)
y_pred_iso_forest = iso_forest.predict(X_test)

# Convert predictions (-1 is anomaly/fraud, 1 is normal)
y_pred_iso_forest = [1 if x == -1 else 0 for x in y_pred_iso_forest]

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred_iso_forest)}")
print(classification_report(y_test, y_pred_iso_forest))