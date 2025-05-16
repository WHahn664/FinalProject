import os

def save_table(results, filepath, tuned_params=None, title=None):
    with open(filepath, "w") as f:
        if title:
            f.write(f"--- {title} ---\n")
        if tuned_params:
            for model_name, params in tuned_params.items():
                f.write(f"Best params for {model_name}: {params}\n")
        f.write(results.to_string())

def save_hyperparams(param_dict, filename):
    with open(filename, 'w') as f:
        for model_name, params in param_dict.items():
            f.write(f"{model_name}: {params}\n")
