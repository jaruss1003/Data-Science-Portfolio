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
