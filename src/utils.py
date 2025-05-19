import os

def save_table(results, filepath, tuned_params=None, title=None):
    """
    This function allows us to save and return a pandas dataframe as a text table.
    It stores it as a.txt file.
    
    Parameters used:
    - results (pd.DataFrame): DataFrame containing the results to save.
    - filepath (str): Path to the output text file.
    - tuned_params (dict): Dictionary of model names to their tuned hyperparameters to include.
    - title (str, optional): Title to write at the top of the file.
    """
    with open(filepath, "w") as f:
        if title:
            f.write(f"--- {title} ---\n")
        if tuned_params:
            for model_name, params in tuned_params.items():
                f.write(f"Best params for {model_name}: {params}\n")
        f.write(results.to_string())

def save_hyperparams(param_dict, filename):
    """
    This function allows us to save and return a dictionary of the model 
    hyperparameters to a text file.

    Parameters used:
    - param_dict (dict): Dictionary mapping model names to their hyperparameter settings.
    - filename (str): Path to the output text file.
    """
    with open(filename, 'w') as f:
        for model_name, params in param_dict.items():
            f.write(f"{model_name}: {params}\n")
