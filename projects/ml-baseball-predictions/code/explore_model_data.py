import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os 


sc = StandardScaler()

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
cleaned_dir = os.path.join(script_dir,'..', 'cleaned_data')

games = pd.read_csv(os.path.join(cleaned_dir, 'stats_for_model.csv'))

# Standardize columns
cols_to_scale = ['ERA_diff', 'BB_diff_pitchers','OBP_diff','errors_diff']
merged_scaled = games.copy()
merged_scaled[cols_to_scale] = sc.fit_transform(merged_scaled[cols_to_scale].fillna(0))

# Correlation with home wins
cols_to_corr = cols_to_scale + ['wins']
corr_matrix = merged_scaled[cols_to_corr].corr()
corr_with_wins = corr_matrix['wins'].drop('wins')

print("\n=== Correlation with Home Team Winning ===")
print(corr_with_wins.sort_values(ascending=False))

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix (Home Team Winning)")
plt.show()
project_root = os.path.join(script_dir, '..')
plots_dir = os.path.join(project_root, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'))