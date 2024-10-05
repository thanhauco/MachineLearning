import sklearn
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load dataset
X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier()

# Define hyperparameter grid for GridSearch
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20]}

# Grid search
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)
print(f'Best parameters (Grid Search): {grid_search.best_params_}')

# Random search
random_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=5, cv=3)
random_search.fit(X, y)
print(f'Best parameters (Random Search): {random_search.best_params_}')
