import pandas as pd

def save_table(df, path):
    fig = df.style.background_gradient(cmap='viridis').to_svg()
    with open(path, "w") as f:
        f.write(fig)

def save_text_block(text, path):
    with open(path, 'w') as f:
        f.write(text)
