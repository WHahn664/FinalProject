"""
These lines of code will make three dictionaries to 
show the parameter grids for hypertuning of regression models.

- Decision Tree Regressor (`param_grid_tree`):
    - `max_depth`: Maximum depth of the tree.
    - `min_samples_split`: Minimum number of samples required to split an internal node.

- Random Forest Regressor (`param_grid_rf`):
    - `n_estimators`: Number of trees in the forest.
    - `max_depth`: Maximum depth of the tree.

- Support Vector Regressor (`param_grid_svr`):
    - `C`: Regularization parameter.
    - `epsilon`: Epsilon in the epsilon-SVR model.
    - `kernel`: Specifies the kernel type to be used in the algorithm (set to 'rbf').

These grids are then searched with GridSearchCV to find the best hyperparameters to use for each of the models
"""
param_grid_tree = {
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None]
}

param_grid_svr = {
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 0.2],
    'kernel': ['rbf']
}
