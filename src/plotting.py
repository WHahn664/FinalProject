import matplotlib.pyplot as plt
import seaborn as sns

def plot_rmse_bars(results_untuned_raw, results_tuned_raw, results_untuned_log, results_tuned_log):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    sns.barplot(x=results_untuned_raw.index, y='Mean RMSE', data=results_untuned_raw, ax=axes[0, 0])
    axes[0, 0].set_title('Untuned Models - Raw Target')
    axes[0, 0].tick_params(axis='x', rotation=45)

    sns.barplot(x=results_tuned_raw.index, y='Mean RMSE', data=results_tuned_raw, ax=axes[0, 1])
    axes[0, 1].set_title('Tuned Models - Raw Target')
    axes[0, 1].tick_params(axis='x', rotation=45)

    sns.barplot(x=results_untuned_log.index, y='Mean RMSE', data=results_untuned_log, ax=axes[1, 0])
    axes[1, 0].set_title('Untuned Models - Log-Transformed Target')
    axes[1, 0].tick_params(axis='x', rotation=45)

    sns.barplot(x=results_tuned_log.index, y='Mean RMSE', data=results_tuned_log, ax=axes[1, 1])
    axes[1, 1].set_title('Tuned Models - Log-Transformed Target')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("results/rmse_comparison.svg", format="svg")
