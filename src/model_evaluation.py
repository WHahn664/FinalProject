import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from .config import param_grid_tree, param_grid_rf, param_grid_svr

def evaluate_untuned(X, y):
    results = {}
    fitted_models = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'SVR': SVR()
    }

    for name, model in models.items():
        rmse_scores = -cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
        r2_scores = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1)

        results[name] = {
            'Mean RMSE': np.mean(rmse_scores),
            'Std RMSE': np.std(rmse_scores),
            'Mean R2': np.mean(r2_scores)
        }

        model.fit(X, y)
        fitted_models[name] = model

    return pd.DataFrame(results).T, fitted_models

def evaluate_tuned(X, y, is_log_target=True):
    results = {}
    fitted_models = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_tree = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid_tree, cv=cv,
                             scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=cv,
                           scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid_svr = GridSearchCV(SVR(), param_grid_svr, cv=cv,
                            scoring='neg_root_mean_squared_error', n_jobs=-1)

    models = {
        'Linear Regression': LinearRegression(),
        'Tuned Decision Tree': grid_tree,
        'Tuned Random Forest': grid_rf,
        'Tuned SVR': grid_svr
    }

    for name, model in models.items():
        rmse_scores = -cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
        r2_scores = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1)

        results[name] = {
            'Mean RMSE': np.mean(rmse_scores),
            'Std RMSE': np.std(rmse_scores),
            'Mean R2': np.mean(r2_scores)
        }

        model.fit(X, y)
        if isinstance(model, GridSearchCV):
            print(f"Best params for {name} ({'log' if is_log_target else 'raw'} target): {model.best_params_}")
            fitted_models[name] = model.best_estimator_
        else:
            fitted_models[name] = model

    return pd.DataFrame(results).T, fitted_models

def evaluate_train_test(models, X_train, y_train_raw, X_test, y_test_raw, label):
    from .utils import save_text_block
    output = f"\n--- Training vs Testing Scores ({label.upper()}) ---\n"
    for name, model in models.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if label == "log":
            y_train_pred = np.expm1(y_train_pred)
            y_test_pred = np.expm1(y_test_pred)

        train_rmse = np.sqrt(mean_squared_error(y_train_raw, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test_raw, y_test_pred))
        train_r2 = r2_score(y_train_raw, y_train_pred)
        test_r2 = r2_score(y_test_raw, y_test_pred)

        output += f"--- {name} ---\n"
        output += f"Train RMSE: {train_rmse:.3f}, Test RMSE: {test_rmse:.3f}\n"
        output += f"Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}\n\n"

    save_text_block(output.strip(), f"results/train_test_scores_{label}.txt")

