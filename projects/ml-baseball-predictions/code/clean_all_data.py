import os
import pandas as pd
import re
import csv  # for quoting constants

# Paths
data_path = 'data/'                 
cleaned_path = 'cleaned_data/'      
os.makedirs(cleaned_path, exist_ok=True)

# Columns where Excel might have stored dates but should be #-#
special_cols = ['H-AB', 'PC-ST', 'home-record', 'away-record']

# Excel row/column limits
EXCEL_MAX_ROWS = 1048576
EXCEL_MAX_COLS = 16384

def recover_dash_format(value):
    if pd.isna(value):
        return "0-0"
    s = str(value).strip()

    # Handle cases like "4-Jan" or "3-March"
    match_alpha = re.match(r'(\d{1,2})-([A-Za-z]+)', s)
    if match_alpha:
        day = int(match_alpha.group(1))
        month_text = match_alpha.group(2)[:3].title()
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        month_num = month_map.get(month_text, 0)
        return f"{month_num}-{day}"

    # Otherwise just grab numbers
    nums = re.findall(r'\d+', s)
    if len(nums) >= 2:
        return f"{int(nums[0])}-{int(nums[1])}"
    elif len(nums) == 1:
        return f"{int(nums[0])}-0"
    else:
        return "0-0"

def read_all(path, filename):
    full = os.path.join(path, filename)
    if filename.endswith('.csv'):
        return pd.read_csv(full, dtype=str, low_memory=False, keep_default_na=False)
    else:
        return pd.read_excel(full, dtype=str)

for file in os.listdir(data_path):
    if not (file.endswith('.csv') or file.endswith('.xlsx')):
        continue

    try:
        df = read_all(data_path, file)

        # Normalize column names: strip whitespace, force hyphens
        df.columns = (
            df.columns
            .str.strip()
            .str.replace(' ', '-', regex=False)
            .str.replace('_', '-', regex=False)
        )

        # Ensure all values are strings
        for col in df.columns:
            df[col] = df[col].astype(str)

        # Apply fix to targeted columns
        for col in special_cols:
            if col in df.columns:
                df[col] = df[col].apply(recover_dash_format)

        # Decide output format based on size
        rows, cols = df.shape
        if rows > EXCEL_MAX_ROWS or cols > EXCEL_MAX_COLS:
            # Too big for Excel → save as CSV with quotes
            cleaned_filename = f'cleaned_{os.path.splitext(file)[0]}.csv'
            df.to_csv(os.path.join(cleaned_path, cleaned_filename),
                      index=False,
                      quoting=csv.QUOTE_ALL)
            print(f"Saved large file as CSV: {cleaned_filename}")
        else:
            # Safe for Excel → save as XLSX
            cleaned_filename = f'cleaned_{os.path.splitext(file)[0]}.xlsx'
            df.to_excel(os.path.join(cleaned_path, cleaned_filename),
                        index=False,
                        engine="openpyxl")
            print(f"Saved small file as XLSX: {cleaned_filename}")

    except Exception as e:
        print(f"Error cleaning {file}: {e}")

print("Full run complete — open files in cleaned_data/ to verify.")
