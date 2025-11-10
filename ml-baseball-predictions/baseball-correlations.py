import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#Loading Datasets
games = pd.read_excel('cleaned_data/cleaned_games.xlsx')
hitters = pd.read_excel('cleaned_data/cleaned_hittersByGame.xlsx')
pitching = pd.read_excel('cleaned_data/cleaned_pitchersByGame.xlsx')

# Ensure dates and extract season
games['Date'] = pd.to_datetime(games['Date'], errors='coerce')
games['season'] = games['Date'].dt.year


#Aggregate Team Stats per Game
hitter = hitters.groupby(['Team','Game'], as_index=False).agg({ 
'OBP': 'mean',
'SLG': 'mean',
'K': 'sum',
'BB': 'sum'
})

pitcher = pitchers.groupby(['Team','Game'], as_index=False).agg({
'ERA': 'mean',
'K': 'sum',
'BB': 'sum'
})

# Create a winner column
games['home_win'] = (games['home-score'] > games['away-score']).astype(int)
games['away_win'] = (games['away-score'] > games['home-score']).astype(int)

# Count total wins for each team per season
home_wins = games.groupby(['season', 'home']).agg({'home_win': 'sum'}).reset_index()
away_wins = games.groupby(['season', 'away']).agg({'away_win': 'sum'}).reset_index()

home_wins.rename(columns={'home_wins': 'Team', 'home_win': 'wins'}, inplace=True)
away_wins.rename(columns={'away_wins': 'Team', 'away_win': 'wins'}, inplace=True)

team_wins = pd.concat([home_wins, away_wins])
team_wins = team_wins.groupby(['season','Team'], as_index=False).agg({'wins':'sum'})

# Merge hitting
team_df = team_wins.merge(hitter, on=['season', 'Team'], how='left')

# Merge pitching
team_df = team_df.merge(pitcher, on=['season', 'Team'], how='left')


# Compute correlation matrix
corr_matrix = team_df.corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix: Team Stats vs Wins")
plt.show()

# Optional: just focus on correlation with wins
corr_with_wins = corr_matrix['wins'].sort_values(ascending=False)
print("Correlation with Wins:\n", corr_with_wins)