# Import necessary libraries
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset (Iris dataset for classification)
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DMatrix for XGBoost (XGBoost's internal data format)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set the XGBoost parameters
# 'objective' determines the kind of problem (multi:softmax is for multi-class classification)
params = {
    'objective': 'multi:softmax',  # Softmax for multi-class classification
    'num_class': 3,                # Number of classes (3 in the Iris dataset)
    'max_depth': 4,                # Maximum depth of a tree
    'eta': 0.3,                    # Learning rate (step size shrinkage)
    'eval_metric': 'mlogloss'      # Evaluation metric (logarithmic loss for classification)
}

# Train the model using the training set
# 'num_boost_round' is the number of boosting iterations (i.e., the number of trees)
bst = xgb.train(params, dtrain, num_boost_round=100)

# Make predictions using the test set
y_pred = bst.predict(dtest)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")