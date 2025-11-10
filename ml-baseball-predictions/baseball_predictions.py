import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


#Loading Datasets
games = pd.read_excel('cleaned_data/cleaned_games.xlsx')
hitters = pd.read_excel('cleaned_data/cleaned_hittersByGame.xlsx')
pitching = pd.read_excel('cleaned_data/cleaned_pitchersByGame.xlsx')

# Ensure dates and extract season
games['date'] = pd.to_datetime(games['date'], errors='coerce')
games['season'] = games['date'].dt.year


#Aggregate Team Stats per Game
hitter = hitters.groupby(['team','game'], as_index=False).agg({ 
'OBP': 'mean'
'SLG': 'mean'
'K': 'sum'
'BB': 'sum'
})

pitcher = pitchers.groupby(['team','game'], as_index=False).agg({
'ERA': 'mean'
'K': 'sum'
'BB': 'sum'
})

# Features and target
X = df[['away-score', 'home-score', 'Total Bases - Away', 'Walks Issued - Home', 'Strikeouts Thrown - Away']]
y = df['winner']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))