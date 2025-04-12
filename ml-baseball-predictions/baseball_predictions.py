import pandas as pd
import os

# Path to data folder
data_path = 'data:/'

# Load all CSVs
files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
dataframes = []

for file in files:
    df = pd.read_csv(os.path.join(data_path, file))
    print(f"\nLoaded: {file} — Shape: {df.shape}")
    print(df.head(3))  # Preview the top 3 rows
    dataframes.append(df)