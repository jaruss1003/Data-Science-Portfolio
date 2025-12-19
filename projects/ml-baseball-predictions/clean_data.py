import pandas as pd
import numpy as np
import os

# Use script's directory as base
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')  # Raw data directory
output_dir = os.path.join(script_dir, 'cleaned_data')  # Cleaned data output directory
os.makedirs(output_dir, exist_ok=True)

# Dynamically find all CSV files in data_dir
all_csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')] if os.path.exists(data_dir) else []
# Filter to only the needed files
needed_files = ['games.csv', 'baserunningNotes.csv', 'fieldingNotes.csv', 'hittersByGame.csv', 'pitchersByGame.csv', 'hittingNotes.csv', 'pitchingNotes.csv', 'hittingHighlights.csv']
csv_files = [f for f in all_csv_files if f in needed_files]
print(f"Found and filtering to needed CSV files in {data_dir}: {csv_files}")

for csv_file in csv_files:
    file_path = os.path.join(data_dir, csv_file)
    df = pd.read_csv(file_path, low_memory=False)  # Added low_memory=False for mixed types
    
    # Drop unnecessary columns (customize as needed) - updated to match exact names and add new ones
    cols_to_drop = ['Umpires', 'postseason info', 'Odds', 'O/U', 'Attendance', 'Capacity', 'Duration', 'WIN - Pitcher - Id', 'LOSS - Pitcher - Id', 'SAVE - Pitcher - Stats', 'SAVE - Pitcher - Id', 'SAVE - Pitcher - Name', 'SAVE - Pitcher - AbbrName', 'SAVE - Pitcher - Record', 'Extra Innings']  # Added pitcher id, save info, extra innings
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # Handle missing values: Impute where appropriate, then drop rows with any missing in kept columns
    if 'games' in csv_file.lower():  # Flexible check for games file
        # Impute numeric essentials (e.g., scores) with median - updated to match exact names
        numeric_cols = ['away-score', 'home-score', 'Walks Issued - Away', 'Walks Issued - Home', 'Stolen Bases - Away', 'Stolen Bases - Home', 'Strikeouts Thrown - Away', 'Strikeouts Thrown - Home', 'Total Bases - Away', 'Total Bases - Home']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        # Drop rows with any missing data in kept columns
        df = df.dropna()
    elif 'hittersByGame' in csv_file.lower():
        # Rename and clean
        df.columns = ['Game', 'Team'] if len(df.columns) == 2 else df.columns
        # Drop rows with any missing data in kept columns
        df = df.dropna()
    elif 'pitchersByGame' in csv_file.lower():
        df['ERA'] = df['ERA'].fillna(df['ERA'].mean())  # Impute ERA
        # Drop rows with any missing data in kept columns
        df = df.dropna()
    else:
        # For other files, drop rows with any missing data in kept columns
        df = df.dropna()
    
    # Note: Keep duplicates as they represent per-team data
    
    # Save cleaned file: CSV for large files (>1M rows), XLSX for small
    output_filename = f"cleaned_{csv_file.replace('.csv', '')}"
    if len(df) > 1000000:  # Excel limit ~1M rows
        output_path = os.path.join(output_dir, f"{output_filename}.csv")
        df.to_csv(output_path, index=False)
        print(f"Cleaned and saved as CSV (large file): {output_path}")
    else:
        output_path = os.path.join(output_dir, f"{output_filename}.xlsx")
        df.to_excel(output_path, index=False)
        print(f"Cleaned and saved as XLSX: {output_path}")

print("Cleaning complete. Update explore_cleaned.py to handle both CSV and XLSX files.")