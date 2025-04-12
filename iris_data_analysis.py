import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Show the first few rows of the dataset
print(iris.head())

# Basic info
print("\nDataset Info:")
print(iris.info())

# Check for missing values
print("\nMissing Values:")
print(iris.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(iris.describe())

# Value counts for species
print("\nSpecies Count:")
print(iris['species'].value_counts())

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Pairplot
sns.pairplot(iris, hue='species')
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(iris.drop(columns='species').corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# iris_data_analysis.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load iris dataset
df = sns.load_dataset('iris')
print(df.head())

# Dataset overview
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

# Plot 2: Pairplot to show relationships
sns.pairplot(df, hue='species')
plt.savefig("pairplot.png")
plt.close()

print("\nPlots saved as 'feature_histograms.png' and 'pairplot.png'")


# 1. Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Splitting the data into features (X) and target variable (y)
X = df.drop('species', axis=1)  # Features
y = df['species']  # Target variable

# Splitting the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Model Building
from sklearn.linear_model import LogisticRegression

# Initialize the logistic regression model
model = LogisticRegression(max_iter=200)

# Train the model on the training data
model.fit(X_train, y_train)

# 3. Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predict the labels for the test set
y_pred = model.predict(X_test)

# Print the classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Splitting the dataset into training and testing sets
X = data.drop('species', axis=1)  # Features (sepal_length, sepal_width, petal_length, petal_width)
y = data['species']  # Target variable (species)

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Importing the Random Forest model
from sklearn.ensemble import RandomForestClassifier

# Initializing and training the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluating the model's accuracy
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Making predictions
y_pred = model.predict(X_test)

# Confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)