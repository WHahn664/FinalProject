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

# These two lines sets the random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# These two lines will load and prepare our data
X, y_raw, y_log = load_data('dataset/forestfires.csv')
X_train, X_test, y_raw_train, y_raw_test, y_log_train, y_log_test = split_data(X, y_raw, y_log)


# This will save the distribution comparison plot as .svg file in a file named "results"
plot_target_distributions(y_raw, y_log, "results/target_distributions.svg")

# These 4 lines of code will evaluate our models
results_untuned_raw, models_untuned_raw = evaluate_untuned(X_train, y_raw_train)
results_untuned_log, models_untuned_log = evaluate_untuned(X_train, y_log_train)
results_tuned_raw, models_tuned_raw, tuned_params_raw = evaluate_tuned(X_train, y_raw_train, is_log_target=False, return_params=True)
results_tuned_log, models_tuned_log, tuned_params_log = evaluate_tuned(X_train, y_log_train, is_log_target=True, return_params=True)

# These 4 lines will save our results as 4 .txt tables in a folder named "results"
save_table(results_untuned_raw, 'results/untuned_raw.txt', title="Untuned models on RAW target")
save_table(results_untuned_log, 'results/untuned_log.txt', title="Untuned models on LOG target")
save_table(results_tuned_raw, 'results/tuned_raw.txt', tuned_params=tuned_params_raw, title="Tuned models on RAW target")
save_table(results_tuned_log, 'results/tuned_log.txt', tuned_params=tuned_params_log, title="Tuned models on LOG target")

# These 4 lines will save our bar plots as 4 .svg files in a folder named "results"
plot_rmse_bars(results_untuned_raw, results_tuned_raw, results_untuned_log, results_tuned_log)
plot_r2_mean_bars(results_untuned_raw, results_tuned_raw, results_untuned_log, results_tuned_log)
plot_r2_std_bars(results_untuned_raw, results_tuned_raw, results_untuned_log, results_tuned_log)
plot_rmse_std_bars(results_untuned_raw, results_tuned_raw, results_untuned_log, results_tuned_log)


# These 4 lines will merge the linear regression model into the tuned raw models and the tuned log models
models_raw_combined = models_tuned_raw.copy()
models_raw_combined["Linear Regression"] = models_untuned_raw["Linear Regression"]
models_log_combined = models_tuned_log.copy()
models_log_combined["Linear Regression"] = models_untuned_log["Linear Regression"]

# This will save our train/test scores as a .txt file in a folder named "results"
save_train_test_scores(models_raw_combined, models_log_combined, X_train, y_raw_train, X_test, y_raw_test,
                       "results/train_test_scores.txt")


