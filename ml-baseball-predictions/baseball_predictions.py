import pandas as pd

# Load your dataset
df = pd.read_csv('cleaned_data/cleaned_games.csv')

# Create target variable: 1 if away team wins, 0 if home team wins
df['winner'] = (df['away-score'] > df['home-score']).astype(int)
print(df[['away-score', 'home-score', 'winner']].head())
# Now you're ready to select features and build a model!
# Check correlations with the winner
correlations = df.corr(numeric_only=True)['winner'].sort_values(ascending=False)
print("Correlation with 'winner':\n", correlations)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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