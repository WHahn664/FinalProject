import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    This function loads allows us to load the forest fire dataset. We are basically applying
    a logarithmic transformation to the area variable for the preprocessing step.

    Describing the parameters:
    filepath (str): This contains the path to the CSV dataset named forestfires.csv

    Returns a dataframe with the log-transformed area variable
    """
    data = pd.read_csv(filepath)
    features = ['temp', 'RH', 'wind', 'rain']
    X = data[features]
    y_raw = data['area']
    y_log = np.log1p(y_raw)
    return X, y_raw, y_log

def split_data(X, y_raw, y_log):
    """
    This function splits the independent variables and the dependent variable 
    into training and testing sets.

    Describing the parameters:
        X (DataFrame): Feature matrix.
        y_raw (Series): Raw target values.
        y_log (Series): Log-transformed target values.

    This is what the function returns:
        X_train (DataFrame): Training feature matrix.
        X_test (DataFrame): Testing feature matrix.
        y_raw_train (Series): Training raw target values.
        y_raw_test (Series): Testing raw target values.
        y_log_train (Series): Training log-transformed target values.
        y_log_test (Series): Testing log-transformed target values.
    """
    X_train, X_test, y_raw_train, y_raw_test = train_test_split(X, y_raw, test_size=0.2, random_state=42)
    _, _, y_log_train, y_log_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    return X_train, X_test, y_raw_train, y_raw_test, y_log_train, y_log_test

