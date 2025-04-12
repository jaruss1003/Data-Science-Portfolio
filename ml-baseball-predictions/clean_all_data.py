import os
import pandas as pd

data_path = 'data:/'
cleaned_path = 'cleaned_data/'

# Create folder for cleaned files if it doesn't exist
os.makedirs(cleaned_path, exist_ok=True)

# Loop through all CSV files
for file in os.listdir(data_path):
    if file.endswith('.csv'):
        print(f'Cleaning: {file}')
        try:
            # Load CSV
            df = pd.read_csv(os.path.join(data_path, file), low_memory=False)

            # Drop fully empty rows/columns
            df.dropna(how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)

            # Drop duplicate rows
            df.drop_duplicates(inplace=True)

            # Attempt to convert columns with mixed types to numeric
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='ignore')
                    except:
                        pass

            # Fill NaNs
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(0, inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)

            # Save cleaned file
            cleaned_filename = f'cleaned_{file}'
            df.to_csv(os.path.join(cleaned_path, cleaned_filename), index=False)
            print(f'Saved cleaned file: {cleaned_filename}\n')

        except Exception as e:
            print(f'Error cleaning {file}: {e}\n')