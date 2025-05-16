import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import os

def evaluate_untuned(X_train, y_train, random_state=42):
    """Evaluate untuned models using 5-fold CV and return results and fitted models on full data."""

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(random_state=random_state),
        "SVR": SVR(),
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        rmses = []
        r2s = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)

            rmses.append(rmse)
            r2s.append(r2)

        results[name] = {
            "Mean RMSE": np.mean(rmses),
            "Std RMSE": np.std(rmses),
            "Mean R2": np.mean(r2s),
            "Std R2": np.std(r2s), 
        }

        # Fit on full training data for later train/test eval
        model.fit(X_train, y_train)
        fitted_models[name] = model

    results_df = pd.DataFrame(results).T
    results_df.index.name = "Model"

    return results_df, fitted_models


def evaluate_tuned(X_train, y_train, is_log_target=False, random_state=42, return_params=False):
    """Tune hyperparameters with GridSearchCV and evaluate with 5-fold CV."""

    param_grids = {
        "Decision Tree": {
            "max_depth": [3, 5, 7, 10],
            "min_samples_split": [2, 5, 10],
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7, 10],
        },
        "SVR": {
            "kernel": ["linear", "rbf"],
            "C": [1, 10, 100],
            "epsilon": [0.1, 0.2, 0.5],
        }
    }

    models = {
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "Random Forest": RandomForestRegressor(random_state=random_state),
        "SVR": SVR(),
    }

    tuned_params = {}
    results = {}
    fitted_models = {}

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, model in models.items():
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=kf,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        tuned_params[name] = grid.best_params_

        # Evaluate with 5-fold CV using best estimator
        rmses = []
        r2s = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            best_model.fit(X_tr, y_tr)
            y_pred = best_model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)

            rmses.append(rmse)
            r2s.append(r2)

        results[name] = {
            "Mean RMSE": np.mean(rmses),
            "Std RMSE": np.std(rmses),
            "Mean R2": np.mean(r2s),
            "Std R2": np.std(r2s), 
        }

        # Fit on full training data for later train/test eval
        best_model.fit(X_train, y_train)
        fitted_models[name] = best_model

    results_df = pd.DataFrame(results).T
    results_df.index.name = "Model"

    if return_params:
        return results_df, fitted_models, tuned_params
    else:
        return results_df, fitted_models


def save_train_test_scores(models_raw, models_log, X_train, y_raw_train, X_test, y_raw_test, output_path):
    """
    Compute and save train/test RMSE and R² scores for raw and log-transformed models.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    lines = []
    lines.append("--- Training vs Testing Scores (RAW) ---\n")
    for name, model in models_raw.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_raw_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_raw_test, y_test_pred))

        train_r2 = r2_score(y_raw_train, y_train_pred)
        test_r2 = r2_score(y_raw_test, y_test_pred)

        lines.append(f"--- {name} ---\n")
        lines.append(f"Train RMSE: {train_rmse:.3f}, Test RMSE: {test_rmse:.3f}\n")
        lines.append(f"Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}\n\n")

    lines.append("--- Training vs Testing Scores (LOG-TRANSFORMED) ---\n")
    for name, model in models_log.items():
        y_log_train_pred = model.predict(X_train)
        y_log_test_pred = model.predict(X_test)

        # Reverse the log transformation
        y_train_pred = np.expm1(y_log_train_pred)
        y_test_pred = np.expm1(y_log_test_pred)

        train_rmse = np.sqrt(mean_squared_error(y_raw_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_raw_test, y_test_pred))

        train_r2 = r2_score(y_raw_train, y_train_pred)
        test_r2 = r2_score(y_raw_test, y_test_pred)

        lines.append(f"--- {name} ---\n")
        lines.append(f"Train RMSE: {train_rmse:.3f}, Test RMSE: {test_rmse:.3f}\n")
        lines.append(f"Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}\n\n")

    with open(output_path, "w") as f:
        f.writelines(lines)
