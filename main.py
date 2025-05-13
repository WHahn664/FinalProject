from src.data_loader import load_data, split_data
from src.model_evaluation import evaluate_untuned, evaluate_tuned, evaluate_train_test
from src.plotting import plot_rmse_bars
from src.utils import save_table

# Load and prepare data
X, y_raw, y_log = load_data('forestfires.csv')
X_train, X_test, y_raw_train, y_raw_test, y_log_train, y_log_test = split_data(X, y_raw, y_log)

# Evaluate models
results_untuned_raw, models_untuned_raw = evaluate_untuned(X_train, y_raw_train)
results_untuned_log, models_untuned_log = evaluate_untuned(X_train, y_log_train)
results_tuned_raw, models_tuned_raw = evaluate_tuned(X_train, y_raw_train, is_log_target=False)
results_tuned_log, models_tuned_log = evaluate_tuned(X_train, y_log_train, is_log_target=True)

# Save results as CSV tables in a new folder called "results".
save_table(results_untuned_raw, 'untuned_raw.csv')
save_table(results_untuned_log, 'untuned_log.csv')
save_table(results_tuned_raw, 'tuned_raw.csv')
save_table(results_tuned_log, 'tuned_log.csv')

# Save RMSE bar plots as a single .svg file in a new folder called "results".
plot_rmse_bars(results_untuned_raw, results_tuned_raw, results_untuned_log, results_tuned_log)

# Save train/test scores as a .txt file in a new folder called "results".
evaluate_train_test(models_tuned_raw, X_train, y_raw_train, X_test, y_raw_test, "raw")
evaluate_train_test(models_tuned_log, X_train, y_raw_train, X_test, y_raw_test, "log")
