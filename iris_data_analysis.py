# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the Iris dataset
df = sns.load_dataset('iris')
print(df.head())

# Dataset Overview
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSummary Statistics:")
print(df.describe())

print("\nSpecies Count:")
print(df['species'].value_counts())

# Plot 1: Histogram of all features
df.hist(figsize=(10, 8), edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.savefig("feature_histograms.png")
plt.close()

# Plot 2: Pairplot
sns.pairplot(df, hue='species')
plt.savefig("pairplot.png")
plt.close()
print("\nPlots saved as 'feature_histograms.png' and 'pairplot.png'")

# Feature & Label Separation
X = df.drop('species', axis=1)  # Features
y = df['species']              # Target

# Train-Test Split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

print("\n[Logistic Regression]")
print("Classification Report:\n", classification_report(y_test, y_pred_log))

conf_matrix_log = confusion_matrix(y_test, y_pred_log)
sns.heatmap(conf_matrix_log, annot=True, fmt="d", cmap="Blues", xticklabels=log_model.classes_, yticklabels=log_model.classes_)
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Model 2: Random Forest (no scaling needed)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_rf, y_train_rf)
y_pred_rf = rf_model.predict(X_test_rf)

print("\n[Random Forest]")
print(f"Model Accuracy: {rf_model.score(X_test_rf, y_test_rf):.2f}")
print("Classification Report:\n", classification_report(y_test_rf, y_pred_rf))

conf_matrix_rf = confusion_matrix(y_test_rf, y_pred_rf)
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Greens", xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot and save the confusion matrix for Random Forest
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_rf, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("confusion_matrix_rf.png")
plt.close()

print("Confusion matrix saved as 'confusion_matrix_rf.png'")