import pandas as pd
df = pd.read_csv('cleaned_data/cleaned_games.csv')
print(df.head(10))
print(df.tail())
print(df.columns)
print(df.describe())
print(df.isnull().sum())
print(df.dtypes)
