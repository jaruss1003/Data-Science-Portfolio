## ML MLB Game Outcome Predictor

This project uses machine learning to predict the outcomes of Major League Baseball (MLB) games based on historical data. The goal is to build a baseline classification model that predicts whether a team will win or lose using features derived from team and game statistics.

## Dataset

The data used in this project comes from publicly available Major League Baseball game logs and statistics.

- Historical MLB game and team-level data (2016–2021)
- Includes game outcomes, team statistics, pitching metrics, and error totals

**Dataset source:**  
https://www.kaggle.com/datasets/josephvm/mlb-game-data

The dataset was cleaned and processed prior to feature engineering and model training.  
All features used in the baseline model were computed using information available **before each game** to avoid data leakage.

## Features

- Data exploration and visualization (correlation heatmap of features vs. home team wins)
- Feature engineering and computation of differentials  
  (e.g., `ERA_diff`, `BB_diff_pitchers`, `OBP_diff`, `errors_diff`)
- Data preprocessing: cleaning, encoding, and handling missing values
- Model training and evaluation using Logistic Regression
- Output of model coefficients to understand feature importance

## Models

- **Logistic Regression** with standardized inputs

Future work:
- Random Forest
- Additional models for improved predictive performance

## Outputs

- Feature importance and model coefficients
- Accuracy and classification report on test set
- Correlation heatmap (`correlation_matrix.png`) of features vs. home team wins

## Getting Started

1. Clone this repository
2. Add the dataset to the `data/` folder
3. Run the scripts in `code/`:

## Clean and validate data
```bash
python3 clean_all_data.py
python3 validate_clean.py ```

## Compute Features 
```bash
python3 compute-features.py ```

## Explore Model Data 
```bash
python3 explore-model-data.py ```

## Train Baseline Model 
```bash
python3 baseball_predictions.py ```

## Status

Baseline Complete — Logistic Regression model predicts home team wins using 2016–2021 data.

Future goals include:
 - Incorporating player-level data and rosters
	- Real-time predictions with updated statistics
	- Additional models and ensemble methods for improved accuracy

## Requirements
	- Python 3.x
	- pandas
	- seaborn
	- matplotlib
	-	scikit-learn