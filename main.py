import random
import numpy as np
from src.data_loader import load_data, split_data
from src.model_evaluation import evaluate_untuned, evaluate_tuned, save_train_test_scores
from src.plotting import (
    plot_rmse_bars,
    plot_r2_mean_bars,
    plot_r2_std_bars,
    plot_rmse_std_bars,
    plot_target_distributions
)
from src.utils import save_table

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load and prepare data
X, y_raw, y_log = load_data('forestfires.csv')
X_train, X_test, y_raw_train, y_raw_test, y_log_train, y_log_test = split_data(X, y_raw, y_log)


# Save distribution comparison plot
plot_target_distributions(y_raw, y_log, "results/target_distributions.svg")

# Evaluate models
results_untuned_raw, models_untuned_raw = evaluate_untuned(X_train, y_raw_train)
results_untuned_log, models_untuned_log = evaluate_untuned(X_train, y_log_train)
results_tuned_raw, models_tuned_raw, tuned_params_raw = evaluate_tuned(X_train, y_raw_train, is_log_target=False, return_params=True)
results_tuned_log, models_tuned_log, tuned_params_log = evaluate_tuned(X_train, y_log_train, is_log_target=True, return_params=True)

# Save results as TXT tables
save_table(results_untuned_raw, 'results/untuned_raw.txt', title="Untuned models on RAW target")
save_table(results_untuned_log, 'results/untuned_log.txt', title="Untuned models on LOG target")
save_table(results_tuned_raw, 'results/tuned_raw.txt', tuned_params=tuned_params_raw, title="Tuned models on RAW target")
save_table(results_tuned_log, 'results/tuned_log.txt', tuned_params=tuned_params_log, title="Tuned models on LOG target")

# Save visualizations as SVG (bar plots only; hyperparams not included)
plot_rmse_bars(results_untuned_raw, results_tuned_raw, results_untuned_log, results_tuned_log)
plot_r2_mean_bars(results_untuned_raw, results_tuned_raw, results_untuned_log, results_tuned_log)
plot_r2_std_bars(results_untuned_raw, results_tuned_raw, results_untuned_log, results_tuned_log)
plot_rmse_std_bars(results_untuned_raw, results_tuned_raw, results_untuned_log, results_tuned_log)


# Merge Linear Regression into tuned_raw models
models_raw_combined = models_tuned_raw.copy()
models_raw_combined["Linear Regression"] = models_untuned_raw["Linear Regression"]

# Save train/test scores to TXT
save_train_test_scores(models_raw_combined, models_tuned_log, X_train, y_raw_train, X_test, y_raw_test,
                       "results/train_test_scores.txt")


