import pandas as pd
import os


# Load datasets
script_dir = os.path.dirname(os.path.abspath(__file__))
cleaned_dir = os.path.join(script_dir, '..', 'cleaned_data')

games = pd.read_excel(os.path.join(cleaned_dir, 'cleaned_games.xlsx'))
hitters = pd.read_excel(os.path.join(cleaned_dir, 'cleaned_hittersByGame.xlsx'))
pitchers = pd.read_excel(os.path.join(cleaned_dir, 'cleaned_pitchersByGame.xlsx'))

# Basic cleanup
for df in [games, hitters, pitchers]:
    df.columns = df.columns.str.strip()
    if 'Team' in df.columns:
        df['Team'] = df['Team'].str.strip()
    if 'home' in df.columns:
        df['home'] = df['home'].str.strip()
    if 'away' in df.columns:
        df['away'] = df['away'].str.strip()


# Compute wins per team per season
games['home_win'] = (games['home-score'] > games['away-score']).astype(int)
games['away_win'] = (games['away-score'] > games['home-score']).astype(int)
games['wins'] = games['home_win']

# Pre-computed team-season hitting stats from TEAM rows
team_rows = hitters[(hitters['Position'] == 'TEAM') & (hitters['Hitters'] == 'TEAM')]

# Home team stats
home_rows = team_rows[team_rows['Team'].isin(games['home'])]
home_stats = home_rows.rename(columns={
    'Team_season_AVG': 'home_AVG',
    'Team_season_OBP': 'home_OBP',
    'Team_season_SLG': 'home_SLG',
    'Team_season_OPS': 'home_OPS'
})[['Game', 'Team', 'home_AVG', 'home_OBP', 'home_SLG', 'home_OPS']]

# Away team stats
away_rows = team_rows[team_rows['Team'].isin(games['away'])]
away_stats = away_rows.rename(columns={
    'Team_season_AVG': 'away_AVG',
    'Team_season_OBP': 'away_OBP',
    'Team_season_SLG': 'away_SLG',
    'Team_season_OPS': 'away_OPS'
})[['Game', 'Team', 'away_AVG', 'away_OBP', 'away_SLG', 'away_OPS']]

# Merge hitting stats
games = games.merge(home_stats, left_on=['Game', 'home'], right_on=['Game', 'Team'], how='left')
games = games.merge(away_stats, left_on=['Game', 'away'], right_on=['Game', 'Team'], how='left')
games.drop(columns=['Team_x','Team_y'], inplace=True)

# Compute hitting differential
games['OBP_diff'] = games['home_OBP'] - games['away_OBP']


# Team-level pitching stats
pitchers_stats = pitchers.groupby(['Team'], as_index=False).agg({
    'ERA': 'mean',
    'BB': 'mean'
}).rename(columns={'BB': 'BB_per_game_pitchers'})
pitchers_stats['ERA'] = pitchers_stats['ERA'].fillna(pitchers_stats['ERA'].mean())

# Merge pitching stats into games
games = games.merge(
    pitchers_stats.rename(columns={
        'Team': 'home',
        'ERA': 'home_ERA',
        'K_per_game_pitchers': 'home_K',
        'BB_per_game_pitchers': 'home_BB'
    }), on='home', how='left'
)

games = games.merge(
    pitchers_stats.rename(columns={
        'Team': 'away',
        'ERA': 'away_ERA',
        'BB_per_game_pitchers': 'away_BB'
    }), on='away', how='left'
)

# Compute pitching differentials
games['ERA_diff'] = games['away_ERA'] - games['home_ERA']
games['BB_diff_pitchers'] = games['away_BB'] - games['home_BB']

# ------------------------------
# Compute errors differential
# ------------------------------
def get_home_errors(row):
    col_name = f"errors_{row['home']}"
    return row[col_name] if col_name in row else 0

def get_away_errors(row):
    col_name = f"errors_{row['away']}"
    return row[col_name] if col_name in row else 0

games['home_errors'] = games.apply(get_home_errors, axis=1)
games['away_errors'] = games.apply(get_away_errors, axis=1)
games['errors_diff'] = games['away_errors'] - games['home_errors']

games.to_csv(os.path.join(cleaned_dir, 'stats_for_model.csv'), index=False)
print("Saved model ready dataset: stats_for_model.csv")

