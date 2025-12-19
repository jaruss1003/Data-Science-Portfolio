import pandas as pd
import numpy as np
import os

def check_data_quality(df, filename):
    """Apply comprehensive data quality checks to a dataframe"""
    print(f"\n{'='*50}")
    print(f"ANALYZING: {filename}")
    print(f"{'='*50}")

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing values found:")
        print(missing[missing > 0])
    else:
        print("\n✅ No missing values")

    # Data types
    print(f"\nData types:")
    print(df.dtypes.value_counts())

    # Numeric columns outlier check
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nOutlier check for {len(numeric_cols)} numeric columns:")
        for col in numeric_cols[:5]:  # Check first 5 to avoid spam
            if df[col].std() > 0:
                outliers = ((df[col] - df[col].mean()) / df[col].std()).abs() > 3
                if outliers.sum() > 0:
                    print(f"  {col}: {outliers.sum()} outliers")

    return df.shape[0], missing.sum()

# Use script's directory as base
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'cleaned_data')

# Dynamically find all cleaned files (CSV or XLSX) in cleaned_data
files = [f for f in os.listdir(data_dir) if f.endswith(('.xlsx', '.csv'))] if os.path.exists(data_dir) else []
print(f"Found files in {data_dir}: {files}")

total_rows = 0
total_missing = 0

for file in files:
    file_path = os.path.join(data_dir, file)
    if file.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file.endswith('.csv'):
        df = pd.read_csv(file_path, low_memory=False)
    rows, missing = check_data_quality(df, file)
    total_rows += rows
    total_missing += missing

print(f"\n{'='*50}")
print("OVERALL SUMMARY")
print(f"{'='*50}")
print(f"Total rows across all files: {total_rows}")
print(f"Total missing values: {total_missing}")

if total_missing == 0:
    print("✅ All files appear clean!")
else:
    print("⚠️  Some data quality issues found - review individual file reports above")