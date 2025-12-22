import pandas as pd
import os
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Model path
script_dir = os.path.dirname(__file__)
cleaned_dir = os.path.join(script_dir,'..', 'cleaned_data')
model_path = os.path.join(script_dir,'logistic_home_win_model.joblib')

#Load Dataset
games = pd.read_csv(os.path.join(cleaned_dir, 'stats_for_model.csv'))

# Features and target
features = ['ERA_diff', 'BB_diff_pitchers', 'OBP_diff', 'errors_diff']
X = games[features]
y = games['wins'] # Home wins: 1, Away wins: 0

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize model
model = LogisticRegression(max_iter=1000, class_weight='balanced')

# Fit on training data
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Feature importance and Coefficients
coeff_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.coef_[0]}).sort_values(by='Coefficient', ascending=False)
print("\nFeature Coefficients:")
print(coeff_df)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save trained model
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")