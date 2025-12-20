import os
import pandas as pd
import numpy as np
import re

# Paths & Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data")
cleaned_path = os.path.join(BASE_DIR, "cleaned_data")
os.makedirs(cleaned_path, exist_ok=True)

needed_files = ['games.csv', 'baserunningNotes.csv', 'fieldingNotes.csv',
                'hittersByGame.csv', 'pitchersByGame.csv',
                'hittingNotes.csv', 'pitchingNotes.csv', 'hittingHighlights.csv']

special_cols = ['H-AB', 'PC-ST', 'home-record', 'away-record']

# Helpers
def recover_dash(value):
    if pd.isna(value): return "0-0"
    nums = re.findall(r'\d+', str(value).strip())
    return f"{nums[0]}-{nums[1]}" if len(nums) >= 2 else f"{nums[0]}-0" if nums else "0-0"

def to_numeric_safe(s, col="", int_id=False):
    cleaned = s.astype(str).str.strip().str.replace(r'[,%]|-+$|\.$', '', regex=True)
    if int_id:
        cleaned = cleaned.str.replace(r'\.0+$', '', regex=True)
    # Convert quietly — non-numeric values become NaN, no warning printed
    return pd.to_numeric(cleaned, errors='coerce')

# Main loop
for csv_file in [f for f in os.listdir(data_path) if f.endswith('.csv') and f in needed_files]:
    file_path = os.path.join(data_path, csv_file)
    filename = csv_file.lower()
    print(f"Processing: {csv_file}")

    df = pd.read_csv(file_path, dtype=str, low_memory=False)
    df.columns = df.columns.str.strip().str.replace(r'[ _]+', '-', regex=True)

    # Dash fix — safe + modern (no applymap)
    existing_special = [c for c in special_cols if c in df.columns]
    if existing_special:
        for col in existing_special:
            df[col] = df[col].map(recover_dash)

    # Drop junk
    drop_cols = ['Umpires', 'postseason-info', 'Odds', 'O/U', 'Attendance', 'Capacity',
                 'Duration', 'Extra-Innings', 'Stadium', 'Location',
                 'SAVE---Pitcher---Stats', 'SAVE---Pitcher---Id', 'SAVE---Pitcher---Name',
                 'SAVE---Pitcher---AbbrName', 'SAVE---Pitcher---Record']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors='ignore')

        # Make Game numeric in fieldingNotes.csv
    if 'fieldingnotes' in filename:
        if 'Game' in df.columns:
            df['Game'] = to_numeric_safe(df['Game'], 'Game')

    if 'hittersbygame' in filename:
        # Numeric cleanup
        cols = ['AB', 'R', 'H', 'RBI', 'BB', 'K', '#P', 'AVG', 'OBP', 'SLG', 'Hitter-Id', 'Game']
        for c in [x for x in cols if x in df.columns]:
            df[c] = to_numeric_safe(df[c], c, int_id=c in ['Hitter-Id', 'Game'])
        df[['AB','R','H','RBI','BB','K','#P','Hitter-Id','Game']] = df[['AB','R','H','RBI','BB','K','#P','Hitter-Id','Game']].fillna(0).astype('Int64')
        df[['AVG','OBP','SLG']] = df[['AVG','OBP','SLG']].fillna(0.0).round(3)

        # Team mask
        team_mask = pd.Series(False, index=df.index)
        if 'Hitters' in df.columns and 'Position' in df.columns:
            h = df['Hitters'].fillna('').str.strip().str.upper()
            p = df['Position'].fillna('').str.strip().str.upper()
            team_mask = (h == 'TEAM') & (p == 'TEAM')
        print(f"  Found {team_mask.sum()} team rows.")

        # hittingNotes for 2B/3B/HR
        notes_path = os.path.join(data_path, 'hittingNotes.csv')
        player_xb = team_xb = pd.DataFrame()
        if os.path.exists(notes_path):
            notes = pd.read_csv(notes_path, dtype=str)
            notes.columns = notes.columns.str.strip().str.replace(r'[ _]+', '-', regex=True)
            notes['Game'] = to_numeric_safe(notes.get('Game', ''), 'Game')
            notes['Stat'] = notes['Stat'].astype(str).str.upper().str.strip()
            notes['last_name'] = notes['Data'].fillna('').str.split(r'\(|,', expand=True)[0].str.strip().str.split().str[-1]

            xb = notes[notes['Stat'].isin(['2B', '3B', 'HR'])]
            if not xb.empty and 'Team' in xb.columns:
                player_xb = xb.groupby(['Game', 'Team', 'last_name'])['Stat'].value_counts().unstack(fill_value=0).reset_index()
                for c in ['2B','3B','HR']: player_xb[c] = player_xb.get(c, 0)
                team_xb = player_xb.groupby(['Game', 'Team'])[['2B','3B','HR']].sum().reset_index()

        # Team aggregation
        individuals = df[~team_mask]
        team_agg = individuals.groupby(['Game', 'Team'], as_index=False)[['AB','H','R','RBI','BB','K','#P']].sum()
        if not team_xb.empty:
            team_agg = team_agg.merge(team_xb, on=['Game','Team'], how='left')
        team_agg[['2B','3B','HR']] = team_agg[['2B','3B','HR']].fillna(0).astype(int)
        team_agg['TB'] = (team_agg['H'] - team_agg[['2B','3B','HR']].sum(axis=1)).clip(0) + \
                         2*team_agg['2B'] + 3*team_agg['3B'] + 4*team_agg['HR']
        team_agg['AVG'] = np.where(team_agg['AB']>0, team_agg['H']/team_agg['AB'], 0.0).round(3)
        team_agg['OBP'] = np.where((team_agg['AB']+team_agg['BB'])>0,
                                   (team_agg['H']+team_agg['BB'])/(team_agg['AB']+team_agg['BB']), 0.0).round(3)
        team_agg['SLG'] = np.where(team_agg['AB']>0, team_agg['TB']/team_agg['AB'], 0.0).round(3)
        team_agg['OPS'] = (team_agg['OBP'] + team_agg['SLG']).round(3)

        # Update team rows
        update = team_agg[['Game','Team','AB','R','H','RBI','BB','K','#P','AVG','OBP','SLG','OPS']]
        df = df.merge(update, on=['Game','Team'], how='left', suffixes=('','_team'))
        for c in ['AB','R','H','RBI','BB','K','#P','AVG','OBP','SLG','OPS']:
            if f'{c}_team' in df.columns:
                df.loc[team_mask, c] = df.loc[team_mask, f'{c}_team']
        df.drop(columns=[c for c in df.columns if '_team' in c], inplace=True, errors='ignore')

        # Individual SLG from notes
        if not player_xb.empty:
            df['last_name'] = df['Hitters'].fillna('').str.split().str[-1]
            df = df.merge(player_xb, on=['Game','Team','last_name'], how='left')
            df[['2B','3B','HR']] = df[['2B','3B','HR']].fillna(0).astype(int)
            df['TB'] = (df['H'] - df[['2B','3B','HR']].sum(axis=1)).clip(0) + \
                       2*df['2B'] + 3*df['3B'] + 4*df['HR']
            df.loc[~team_mask, 'SLG'] = np.where(df.loc[~team_mask, 'AB']>0,
                                                 df.loc[~team_mask, 'TB']/df.loc[~team_mask, 'AB'], 0.0).round(3)
            df.drop(columns=['last_name','2B','3B','HR','TB'], inplace=True, errors='ignore')

        # OPS for everyone
        df['OPS'] = (df['OBP'].fillna(0.0) + df['SLG'].fillna(0.0)).round(3)

                        # Add Season & Date from games.csv + cumulative team season stats
        games_path = os.path.join(data_path, 'games.csv')
        if os.path.exists(games_path):
            games = pd.read_csv(games_path, dtype=str)
            games.columns = games.columns.str.strip().str.replace(r'[ _]+', '-', regex=True)
            games['Game'] = to_numeric_safe(games.get('Game', ''), 'Game')
            if 'Date' in games.columns:
                games['Date'] = pd.to_datetime(games['Date'], errors='coerce')
                games['Season'] = games['Date'].dt.year
            df = df.merge(games[['Game','Season','Date']], on='Game', how='left')

            # Cumulative season stats for team rows (no TB column required)
            if 'Season' in df.columns and 'Date' in df.columns:
                team_mask = (
                    df['Hitters'].fillna('').str.strip().str.upper() == 'TEAM'
                ) & (
                    df['Position'].fillna('').str.strip().str.upper() == 'TEAM'
                )
                if team_mask.any():
                    team_df = df[team_mask].copy()

                    # Approximate today's TB from accurate SLG (very close and avoids extra columns)
                    team_df['today_TB'] = (team_df['SLG'] * team_df['AB']).round(0).astype(int)

                    team_df = team_df.sort_values(['Season', 'Team', 'Date'])
                    for stat in ['H', 'AB', 'BB', 'today_TB']:
                        team_df[f'cum_{stat}'] = team_df.groupby(['Season', 'Team'])[stat].cumsum().shift(1).fillna(0)

                    team_df['season_AVG'] = np.where(team_df['cum_AB'] > 0,
                                                     team_df['cum_H'] / team_df['cum_AB'], 0.0).round(3)
                    team_df['season_OBP'] = np.where((team_df['cum_AB'] + team_df['cum_BB']) > 0,
                                                     (team_df['cum_H'] + team_df['cum_BB']) / (team_df['cum_AB'] + team_df['cum_BB']), 0.0).round(3)
                    team_df['season_SLG'] = np.where(team_df['cum_AB'] > 0,
                                                     team_df['cum_today_TB'] / team_df['cum_AB'], 0.0).round(3)
                    team_df['season_OPS'] = (team_df['season_OBP'] + team_df['season_SLG']).round(3)

                    df = df.merge(team_df[['Game', 'Team', 'season_AVG', 'season_OBP', 'season_SLG', 'season_OPS']],
                                  on=['Game', 'Team'], how='left')
    elif 'pitchersbygame' in filename:
        cols = ['IP','H','R','ER','BB','K','HR','PC','ERA','Pitcher-Id','Game']
        for c in [x for x in cols if x in df.columns]:
            df[c] = to_numeric_safe(df[c], c)
        if 'IP' in df.columns:
            df['IP'] = df['IP'].fillna(0.0).round(2)
            df = df[df['IP'] > 0]
        df[['H','R','ER','BB','K','HR','PC','Pitcher-Id','Game']] = df[['H','R','ER','BB','K','HR','PC','Pitcher-Id','Game']].fillna(0).astype('Int64')

    elif 'games' in filename:
        cols = ['away-score', 'home-score','Stolen-Bases-Away', 'Stolen-Bases-Home', 'Strikeouts-Thrown-Away','Strikeouts-Thrown-Home', 'Total-Bases-Away', 'Total-Bases-Home', 'Game','Walks-Issued---Away', 'Walks-Issued---Home', 'Stolen-Bases---Away', 'Stolen-Bases---Home', 'Strikeouts-Thrown---Away', 'Strikeouts-Thrown---Home', 'Total-Bases---Away', 'Total-Bases---Home','WIN---Pitcher---Id','LOSS---Pitcher---Id']
        for c in [x for x in cols if x in df.columns]:
            df[c] = to_numeric_safe(df[c]).fillna(0)
        if 'Date' in df.columns:
            df['Season'] = pd.to_datetime(df['Date'], errors='coerce').dt.year

        # Add fielding errors from fieldingNotes.csv
        fielding_path = os.path.join(data_path, 'fieldingNotes.csv')
        if os.path.exists(fielding_path) and 'Game' in df.columns:
            df['Game'] = to_numeric_safe(df['Game'], 'Game')
            fielding = pd.read_csv(fielding_path, dtype=str)
            fielding.columns = fielding.columns.str.strip().str.replace(r'[ _]+', '-', regex=True)
            fielding['Game'] = to_numeric_safe(fielding.get('Game', ''), 'Game')
            fielding['Stat'] = fielding['Stat'].astype(str).str.upper().str.strip()

            errors = fielding[fielding['Stat'] == 'E']
            if not errors.empty and 'Team' in errors.columns:
                error_counts = errors.groupby(['Game', 'Team']).size().reset_index(name='errors')
                # Pivot to get home and away errors
                error_pivot = error_counts.pivot(index='Game', columns='Team', values='errors').fillna(0).astype(int)
                # Rename columns safely (use team abbr if available, else generic)
                error_pivot.columns = [f'errors_{col}' for col in error_pivot.columns]
                error_pivot = error_pivot.reset_index()

                df = df.merge(error_pivot, on='Game', how='left')
                df['errors_home'] = df.get('errors_home', 0)
                df['errors_away'] = df.get('errors_away', 0)

       # Final cleanup & save
    df.dropna(subset=[c for c in df.columns if c not in special_cols], inplace=True)

    # Fix timezone issue in Date column before saving to Excel
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Date'] = df['Date'].dt.tz_localize(None)  # Will do nothing if no timezone
    out = f"cleaned_{os.path.splitext(csv_file)[0]}{'.csv' if len(df)>1_000_000 else '.xlsx'}"
    out_path = os.path.join(cleaned_path, out)
    if out.endswith('.csv'):
        df.to_csv(out_path, index=False)
    else:
        df.to_excel(out_path, index=False)
    print(f"  Saved → {out_path}\n")
print("All done!")