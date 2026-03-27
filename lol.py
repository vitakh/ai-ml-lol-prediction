# import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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
# merging the team match data with the match data to get the rank for each match
df_base = team_match_data.merge(match_data[['MatchId', 'RankFk']], left_on='MatchFk', right_on='MatchId', how='left')
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
feature_cols = ['BaronKills', 'RiftHeraldKills', 'DragonKills', 'TowerKills', 'Kills', 'RankFk']
X = df_base[feature_cols]
y = df_base['Win']

# logistic regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# predicting on the test set
y_pred = model.predict(X_test)

# evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy score for logistic regression model: {accuracy:.4f}')

# predicting win loss based on individuals performance
# baseline creation for individual performance, features from MatchStatsTbl
features = ['kills', 'deaths', 'assists', 'DmgDealt', 'DmgTaken', 'TotalGold', 'MinionsKilled']

X_individual = match_stats_data[features]
y_individual = match_stats_data['Win']

# logistic regression for individual performance
X_train_ind, X_test_ind, y_train_ind, y_test_ind = train_test_split(X_individual, y_individual, test_size=0.2, random_state=42)
model_individual = LogisticRegression(max_iter=1000)
model_individual.fit(X_train_ind, y_train_ind)

y_pred_ind = model_individual.predict(X_test_ind)
accuracy_ind = accuracy_score(y_test_ind, y_pred_ind)
print(f'Accuracy score for logistic regression model based on individual performance: {accuracy_ind:.4f}')

# lane matchup analysis

# for user input, make sure to put the id of the champion (convert supplied name into id)
# merging the match stats data with the summoner match data to get the champion
df_lane = match_stats_data.merge(summoner_data[['SummonerMatchId', 'ChampionFk']], left_on='SummonerMatchFk', right_on='SummonerMatchId', how='left')
# cleaning after merge
df_lane = df_lane.drop(columns=['SummonerMatchId', 'SummonerMatchFk'])
df_lane.dropna(inplace=True)
print(df_lane.head())

# win column: boolean for dmg taken > dmg dealt (assuming that if a player takes more damage than they deal, they are likely to lose)
df_lane['Win'] = df_lane['DmgTaken'] <= df_lane['DmgDealt']

# baseline creation
feature_lane = ['ChampionFk', 'EnemyChampionFk']
X_lane = df_lane[feature_lane]
y_lane = df_lane['Win']

# logistic regression
X_train_lane, X_test_lane, y_train_lane, y_test_lane = train_test_split(X_lane, y_lane, test_size=0.2, random_state=42)
model_lane = LogisticRegression(max_iter=1000)
model_lane.fit(X_train_lane, y_train_lane)

# predicting on the test set
y_pred_lane = model_lane.predict(X_test_lane)

# evaluating the model
accuracy_lane = accuracy_score(y_test_lane, y_pred_lane)
print(f'Accuracy score for logistic regression model for lane matchups: {accuracy_lane:.4f}')
