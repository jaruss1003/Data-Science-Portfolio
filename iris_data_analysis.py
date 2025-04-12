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
