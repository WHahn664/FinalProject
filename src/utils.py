import pandas as pd
import os
        
def save_table(df, filename):
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(df.to_string(index=True))

def save_text_block(text, path):
    with open(path, 'w') as f:
        f.write(text)
