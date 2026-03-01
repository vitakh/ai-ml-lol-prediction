# import libraries
import numpy as np
import pandas as pd

# data preprocessing
# load the champion data
champ_data = pd.read_csv('kaggle/ChampionTbl.csv')
print(champ_data.head())
# load the item data
item_data = pd.read_csv('kaggle/ItemTbl.csv')
print(item_data.head())
# load the item data
match_stats_data = pd.read_csv('kaggle/MatchStatsTbl.csv')
print(match_stats_data.head())
# load the match data
match_data = pd.read_csv('kaggle/MatchTbl.csv')
print(match_data.head())
# load the rank data
rank_data = pd.read_csv('kaggle/RankTbl.csv')
print(rank_data.head())
# load the summoner data
summoner_data = pd.read_csv('kaggle/SummonerMatchTbl.csv')
print(summoner_data.head())
# load the team match data
team_match_data = pd.read_csv('kaggle/TeamMatchTbl.csv')
print(team_match_data.head())

# data cleaning
og_row_count = match_data.shape[0]
match_data = match_data.dropna()
dropped_row_count = og_row_count - match_data.shape[0]
print(f'Dropped {dropped_row_count} rows from match table with missing values.')

# checking if any row has no winner in the team match data
print((team_match_data['BlueWin'] + team_match_data['RedWin']).value_counts(dropna=False))
# dropping rows with no winner
team_match_data = team_match_data[(team_match_data['BlueWin'] + team_match_data['RedWin'] == 1)].reset_index(drop=True)

# data merging for logistic regression to predict win/loss
# organizing the team match table and rank
# need only one column for blue win (the data for red and blue is inverse of each other)
team_match_data['Win'] = team_match_data['BlueWin']  
# merging the team match data with the match data to get the rank and game duration for each match
df_base = team_match_data.merge(match_data[['MatchId', 'RankFk', 'GameDuration']], left_on='MatchFk', right_on='MatchId', how='left')
# cleaning after merge
df_base = df_base.drop(columns=['BlueWin', 'RedWin', 'TeamID', 'MatchId'])
print(df_base.head())

# differential feature creation
df_base['BaronKills'] = df_base['BlueBaronKills'] - df_base['RedBaronKills']
df_base['RiftHeraldKills'] = df_base['BlueRiftHeraldKills'] - df_base['RedRiftHeraldKills']
df_base['DragonKills'] = df_base['BlueDragonKills'] - df_base['RedDragonKills']
df_base['TowerKills'] = df_base['BlueTowerKills'] - df_base['RedTowerKills']
df_base['Kills'] = df_base['BlueKills'] - df_base['RedKills']

# baseline creation
feature_cols = ['BaronKills', 'RiftHeraldKills', 'DragonKills', 'TowerKills', 'Kills', 'GameDuration', 'RankFk']
X = df_base[feature_cols]
y = df_base['Win']

# logistic regression