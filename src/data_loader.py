import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filepath):
    data = pd.read_csv(filepath)
    features = ['temp', 'RH', 'wind', 'rain']
    X = data[features]
    y_raw = data['area']
    y_log = np.log1p(y_raw)
    return X, y_raw, y_log

def split_data(X, y_raw, y_log):
    X_train, X_test, y_raw_train, y_raw_test = train_test_split(X, y_raw, test_size=0.2, random_state=42)
    _, _, y_log_train, y_log_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    return X_train, X_test, y_raw_train, y_raw_test, y_log_train, y_log_test
