import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_target_distributions(y_raw, y_log, output_path):
    """
    Plot the distribution of original and log-transformed target values and save as SVG.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(y_raw, bins=30, kde=True)
    plt.title('Original Area Distribution')

    plt.subplot(1, 2, 2)
    sns.histplot(y_log, bins=30, kde=True)
    plt.title('Log-Transformed Area Distribution')

    plt.tight_layout()
    plt.savefig(output_path, format='svg')
    plt.close()


def plot_rmse_bars(results_untuned_raw, results_tuned_raw, results_untuned_log, results_tuned_log):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

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
    plt.close()


def plot_r2_mean_bars(results_untuned_raw, results_tuned_raw, results_untuned_log, results_tuned_log):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.barplot(x=results_untuned_raw.index, y='Mean R2', data=results_untuned_raw, ax=axes[0, 0])
    axes[0, 0].set_title('Untuned Models - Raw Target')
    axes[0, 0].tick_params(axis='x', rotation=45)

    sns.barplot(x=results_tuned_raw.index, y='Mean R2', data=results_tuned_raw, ax=axes[0, 1])
    axes[0, 1].set_title('Tuned Models - Raw Target')
    axes[0, 1].tick_params(axis='x', rotation=45)

    sns.barplot(x=results_untuned_log.index, y='Mean R2', data=results_untuned_log, ax=axes[1, 0])
    axes[1, 0].set_title('Untuned Models - Log-Transformed Target')
    axes[1, 0].tick_params(axis='x', rotation=45)

    sns.barplot(x=results_tuned_log.index, y='Mean R2', data=results_tuned_log, ax=axes[1, 1])
    axes[1, 1].set_title('Tuned Models - Log-Transformed Target')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("results/r2_mean_comparison.svg", format="svg")
    plt.close()


def plot_r2_std_bars(results_untuned_raw, results_tuned_raw, results_untuned_log, results_tuned_log):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.barplot(x=results_untuned_raw.index, y='Std R2', data=results_untuned_raw, ax=axes[0, 0])
    axes[0, 0].set_title('Untuned Models - Raw Target')
    axes[0, 0].tick_params(axis='x', rotation=45)

    sns.barplot(x=results_tuned_raw.index, y='Std R2', data=results_tuned_raw, ax=axes[0, 1])
    axes[0, 1].set_title('Tuned Models - Raw Target')
    axes[0, 1].tick_params(axis='x', rotation=45)

    sns.barplot(x=results_untuned_log.index, y='Std R2', data=results_untuned_log, ax=axes[1, 0])
    axes[1, 0].set_title('Untuned Models - Log-Transformed Target')
    axes[1, 0].tick_params(axis='x', rotation=45)

    sns.barplot(x=results_tuned_log.index, y='Std R2', data=results_tuned_log, ax=axes[1, 1])
    axes[1, 1].set_title('Tuned Models - Log-Transformed Target')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("results/r2_std_comparison.svg", format="svg")
    plt.close()



def plot_rmse_std_bars(results_untuned_raw, results_tuned_raw, results_untuned_log, results_tuned_log):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.barplot(x=results_untuned_raw.index, y='Std RMSE', data=results_untuned_raw, ax=axes[0, 0])
    axes[0, 0].set_title('Untuned Models - Raw Target')
    axes[0, 0].tick_params(axis='x', rotation=45)

    sns.barplot(x=results_tuned_raw.index, y='Std RMSE', data=results_tuned_raw, ax=axes[0, 1])
    axes[0, 1].set_title('Tuned Models - Raw Target')
    axes[0, 1].tick_params(axis='x', rotation=45)

    sns.barplot(x=results_untuned_log.index, y='Std RMSE', data=results_untuned_log, ax=axes[1, 0])
    axes[1, 0].set_title('Untuned Models - Log-Transformed Target')
    axes[1, 0].tick_params(axis='x', rotation=45)

    sns.barplot(x=results_tuned_log.index, y='Std RMSE', data=results_tuned_log, ax=axes[1, 1])
    axes[1, 1].set_title('Tuned Models - Log-Transformed Target')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig("results/rmse_std_comparison.svg", format="svg")
    plt.close()
