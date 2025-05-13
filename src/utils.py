import pandas as pd
import os
        
def save_table(df, filename, results_dir='results'):
    """
    Saves a DataFrame as a .csv file in the results directory.
    """
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    # Save the dataframe as a CSV file
    df.to_csv(filepath, index=False)
    print(f"Table saved to {filepath}")

def save_text_block(text, path):
    with open(path, 'w') as f:
        f.write(text)
