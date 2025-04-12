# Data Science Portfolio

This repository is where I will upload projects related to Data Science and AI. Currently, I am working on Python-based analysis and machine-learning models.

**Technologies used:** Python, Pandas, NumPy, Scikit-learn

---

## Projects

# Iris Dataset Analysis

This project is a beginner-friendly data science analysis of the classic **Iris flower dataset**. It includes data exploration, visualization, and basic machine learning models to classify flower species based on measurements.

## Project Goals

- Perform exploratory data analysis (EDA) on the Iris dataset.
- Visualize feature relationships and distributions.
- Train and evaluate machine learning models for classification.

## Contents

- `iris_data_analysis.py` — The main script containing all analysis steps.
- `feature_histograms.png` — Histogram of each feature.
- `pairplot.png` — Pairplot showing feature relationships by species.
- `confusion_matrix_rf.png` — Confusion matrix from Random Forest classifier.

## Steps Performed

1. **Data Loading and Exploration**
   - Used Seaborn’s built-in Iris dataset.
   - Checked dataset structure, missing values, and basic statistics.

2. **Data Visualization**
   - Created histograms and pairplots to understand feature distributions.
   - Generated a heatmap to show feature correlations.

3. **Modeling**
   - **Logistic Regression**
     - Scaled features using `StandardScaler`.
     - Evaluated performance with classification report and confusion matrix.
   - **Random Forest Classifier**
     - No feature scaling required.
     - Visualized predictions with a confusion matrix.
     - Achieved high classification accuracy.

## How to Run

1. Clone the repository or download the files.
2. Make sure you have Python and the required libraries:
   ```bash
   pip install pandas seaborn matplotlib scikit-learn
