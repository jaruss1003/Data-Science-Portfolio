import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# ------------------------------
# Load datasets
# ------------------------------
games = pd.read_excel('cleaned_data/cleaned_games.xlsx')
hitters = pd.read_excel('cleaned_data/cleaned_hittersByGame.xlsx')
pitchers = pd.read_excel('cleaned_data/cleaned_pitchersByGame.xlsx')

# ------------------------------
# Basic cleanup
# ------------------------------
# Strip whitespace from column names and team-related columns
for df in [games, hitters, pitchers]:
    df.columns = df.columns.str.strip()
    if 'Team' in df.columns:
        df['Team'] = df['Team'].str.strip()
    if 'home' in df.columns:
        df['home'] = df['home'].str.strip()
    if 'away' in df.columns:
        df['away'] = df['away'].str.strip()

# Ensure numeric columns are properly cleaned and converted
numeric_columns_hitters = ['OBP', 'SLG', 'K']
numeric_columns_pitchers = ['ERA', 'K', 'BB']

for col in numeric_columns_hitters:
    hitters[col] = pd.to_numeric(hitters[col], errors='coerce')
for col in numeric_columns_pitchers:
    pitchers[col] = pd.to_numeric(pitchers[col], errors='coerce')

# Handle invalid ERA values in pitchers
pitchers['ERA'].replace([float('inf'), -float('inf')], pd.NA, inplace=True)

# ------------------------------
# Compute wins per team per season
# ------------------------------
# Compare scores to determine home and away wins
games['home_win'] = (games['home-score'] > games['away-score']).astype(int)
games['away_win'] = (games['away-score'] > games['home-score']).astype(int)

# Aggregate home wins by team and season
home_wins = games.groupby(['season', 'home'], as_index=False)['home_win'].sum()
home_wins.rename(columns={'home': 'Team', 'home_win': 'wins'}, inplace=True)

# Aggregate away wins by team and season
away_wins = games.groupby(['season', 'away'], as_index=False)['away_win'].sum()
away_wins.rename(columns={'away': 'Team', 'away_win': 'wins'}, inplace=True)

# Combine home and away wins to get total wins per team
wins_df = pd.concat([home_wins, away_wins]).groupby(['season', 'Team'], as_index=False)['wins'].sum()

# ------------------------------
# Team-level hitting stats
# ------------------------------
hitters_stats = hitters.groupby(['Team'], as_index=False).agg({
    'OBP': 'mean',
    'SLG': 'mean',
    'K': 'mean'
}).rename(columns={'K': 'K_per_game_hitters'})

# ------------------------------
# Team-level pitching stats
# ------------------------------
pitchers_stats = pitchers.groupby(['Team'], as_index=False).agg({
    'ERA': 'mean',
    'K': 'mean',
    'BB': 'mean'
}).rename(columns={'K': 'K_per_game_pitchers', 'BB': 'BB_per_game_pitchers'})

# Handle NaN ERA values by replacing with the mean ERA
pitchers_stats['ERA'] = pitchers_stats['ERA'].fillna(pitchers_stats['ERA'].mean())

# ------------------------------
# Merge team stats
# ------------------------------
team_df = wins_df.merge(hitters_stats, on='Team', how='left').merge(pitchers_stats, on='Team', how='left')

# ------------------------------
# Prepare opponent stats for differentials
# ------------------------------
opponents = team_df.copy()
opponents.columns = ['season', 'Opponent'] + [f"opp_{col}" for col in opponents.columns[2:]]

# Merge games to get team vs opponent stats
merged = (
    games[['season', 'home', 'away']]
    .merge(team_df, left_on=['season', 'home'], right_on=['season', 'Team'], how='left')
    .merge(opponents, left_on=['season', 'away'], right_on=['season', 'Opponent'], how='left')
)

# Compute differentials (team - opponent, or opponent - team for pitching stats)
merged['OBP_diff'] = merged['OBP'] - merged['opp_OBP']
merged['SLG_diff'] = merged['SLG'] - merged['opp_SLG']
merged['ERA_diff'] = merged['opp_ERA'] - merged['ERA']  # Lower ERA is better
merged['K_diff_hitters'] = merged['opp_K_per_game_hitters'] - merged['K_per_game_hitters']
merged['K_diff_pitchers'] = merged['K_per_game_pitchers'] - merged['opp_K_per_game_pitchers']
merged['BB_diff_pitchers'] = merged['opp_BB_per_game_pitchers'] - merged['BB_per_game_pitchers']

# Average differentials per team per season
diffs = merged.groupby(['season', 'Team']).mean(numeric_only=True).reset_index()

# ------------------------------
# Merge wins back to the differentials DataFrame
# ------------------------------
diffs = diffs.merge(wins_df, on=['season', 'Team'], how='left')

# Handle duplicate columns if present
if 'wins_x' in diffs.columns and 'wins_y' in diffs.columns:
    diffs.rename(columns={'wins_y': 'wins'}, inplace=True)
    diffs.drop(columns=['wins_x'], inplace=True)

# Ensure the wins column exists and is valid
if 'wins' not in diffs.columns:
    raise KeyError("The 'wins' column is missing in the differentials DataFrame after merge.")

# ------------------------------
# Standardize Data and Correlate with wins
# ------------------------------
cols_to_scale = [
    'OBP_diff', 'SLG_diff', 'ERA_diff',
    'K_diff_hitters', 'K_diff_pitchers', 'BB_diff_pitchers'
]

diffs_scaled = diffs.copy()
diffs_scaled[cols_to_scale] = sc.fit_transform(diffs_scaled[cols_to_scale])

cols_to_corr = cols_to_scale + ['wins']

# Compute correlation matrix
corr_matrix = diffs_scaled[cols_to_corr].corr()
corr_with_wins = corr_matrix['wins'].drop('wins')

# Print correlation results
print("\n=== Correlation with Wins (Differentials) ===")
print(corr_with_wins)

# ------------------------------
# Heatmap of correlations
# ------------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix (Differentials + Wins)")
plt.show()
