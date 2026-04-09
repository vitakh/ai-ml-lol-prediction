# import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def preprocess_and_train():
    """
    Preprocesses the data and trains logistic regression models for team performance, individual performance, and lane matchup win/loss prediction.

    Returns:
        model_team: trained logistic regression model for team performance prediction.
        model_individual: trained logistic regression model for individual performance prediction.
        model_lane: Trained trained regression model for lane matchup prediction.
    """
    # data preprocessing
    # load the item data
    match_stats_data = pd.read_csv('kaggle/MatchStatsTbl.csv')
    # load the match data
    match_data = pd.read_csv('kaggle/MatchTbl.csv')
    # load the summoner data
    summoner_data = pd.read_csv('kaggle/SummonerMatchTbl.csv')
    # load the team match data
    team_match_data = pd.read_csv('kaggle/TeamMatchTbl.csv')

    # data cleaning
    match_data = match_data.dropna()

    # checking if any row has no winner in the team match data
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

    # feature creation
    df_base['BaronKills'] = df_base['BlueBaronKills']
    df_base['RiftHeraldKills'] = df_base['BlueRiftHeraldKills']
    df_base['DragonKills'] = df_base['BlueDragonKills']
    df_base['TowerKills'] = df_base['BlueTowerKills']
    df_base['Kills'] = df_base['BlueKills']

    # baseline creation
    feature_cols = ['BaronKills', 'RiftHeraldKills', 'DragonKills', 'TowerKills', 'Kills', 'RankFk']
    X = df_base[feature_cols]
    y = df_base['Win']

    # logistic regression for the team performance model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_team = LogisticRegression(max_iter=1000)
    model_team.fit(X_train, y_train)

    # predicting on the test set
    y_pred = model_team.predict(X_test)

    # evaluating the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy score for team performance model: {accuracy:.4f}')

    # predicting win/loss based on individuals performance
    # baseline creation for individual performance, features from MatchStatsTbl
    features = ['kills', 'deaths', 'assists', 'TotalGold', 'MinionsKilled']

    X_individual = match_stats_data[features]
    y_individual = match_stats_data['Win']

    # logistic regression for individual performance
    X_train_ind, X_test_ind, y_train_ind, y_test_ind = train_test_split(X_individual, y_individual, test_size=0.2, random_state=42)
    model_individual = LogisticRegression(max_iter=1000)
    model_individual.fit(X_train_ind, y_train_ind)
    
    y_pred_ind = model_individual.predict(X_test_ind)
    accuracy_ind = accuracy_score(y_test_ind, y_pred_ind)
    print(f'Accuracy score for individual performance model: {accuracy_ind:.4f}')

    # lane matchup analysis
    # for user input, make sure to put the id of the champion (convert supplied name into id)
    # merging the match stats data with the summoner match data to get the champion
    df_lane = match_stats_data.merge(summoner_data[['SummonerMatchId', 'ChampionFk']], left_on='SummonerMatchFk', right_on='SummonerMatchId', how='left')
    # cleaning after merge
    df_lane = df_lane.drop(columns=['SummonerMatchId', 'SummonerMatchFk'])
    df_lane.dropna(inplace=True)
    # win column: boolean for dmg taken > dmg dealt (assuming that if a player takes more damage than they deal, they are likely to lose)
    df_lane['Win'] = df_lane['DmgTaken'] <= df_lane['DmgDealt']

    # baseline creation
    feature_lane = ['ChampionFk', 'EnemyChampionFk']
    X_lane = df_lane[feature_lane]
    y_lane = df_lane['Win']

    # logistic regression
    X_train_lane, X_test_lane, y_train_lane, y_test_lane = train_test_split(X_lane, y_lane, test_size=0.2, random_state=42)
    model_lane = LogisticRegression(max_iter=1000, class_weight='balanced')
    model_lane.fit(X_train_lane, y_train_lane)

    # predicting on the test set
    y_pred_lane = model_lane.predict(X_test_lane)

    # evaluating the model
    accuracy_lane = accuracy_score(y_test_lane, y_pred_lane)
    print(f'Accuracy score for lane matchup model: {accuracy_lane:.4f}')

    return model_team, model_individual, model_lane


def lane_matchup(model_lane):
    """
    Predicts the result of the lane matchup based on a user inputs of names of the champion and the enemy champion you are playing against.

    Args: 
        model_lane: trained logistic regression model for lane matchup prediction.
    
    Returns:
        prints the predicted outcome (win/loss) of the current lane matchup.
    """
    print("Enter the name of the champion you are playing (e.g. 'Ahri'):")
    # parse your champion
    champion_name = input("Your Champion Name: ").strip()
    # convert champion name to champion id
    champ_data = pd.read_csv('kaggle/ChampionTbl.csv')
    champion_row = champ_data[champ_data['ChampionName'].str.lower() == champion_name.lower()]
    if champion_row.empty:
        print("Champion not found. Please check the name and try again.")
        return
    ChampionFk = champion_row['ChampionId'].values[0]
    
    # parse your enemy
    print("Enter the name of the enemy champion you are laning against (e.g. 'Zed'):")
    enemy_champion_name = input("Enemy Champion Name: ").strip()
    enemy_champion_row = champ_data[champ_data['ChampionName'].str.lower() == enemy_champion_name.lower()]
    if enemy_champion_row.empty:
        print("Enemy champion not found. Please check the name and try again.")
        return
    EnemyChampionFk = enemy_champion_row['ChampionId'].values[0]
    
    # predict the lane matchup win/loss
    input_df = pd.DataFrame([[ChampionFk, EnemyChampionFk]], columns=['ChampionFk', 'EnemyChampionFk'])
    y_pred_lane = model_lane.predict(input_df)
    if y_pred_lane[0] == 1:
        print(f"The model predicts that your {champion_name} are likely to WIN this lane matchup against {enemy_champion_name}.")
    else:
        print(f"The model predicts that your {champion_name} are likely to LOSE this lane matchup against {enemy_champion_name}.")


def ind_performance(model_individual):
    """
    Predicts the result of the individual performance based on a user inputs of match stats, such as kills, deaths, assists, total gold, and minions killed.
    
    Args: 
        model_individual: trained logistic regression model for individual performance prediction.
    
    Returns:
        prints the predicted outcome (win/loss) of the individual performance.
    """

    print("Enter your performance stats for the match:")
    # get the features if valid, if not, clarify and ask again
    try:
        kills = int(input("Kills: ").strip())
        deaths = int(input("Deaths: ").strip())
        assists = int(input("Assists: ").strip())
        total_gold = int(input("Total Gold: ").strip())
        minions_killed = int(input("Minions Killed: ").strip())
    except ValueError:
        print("Invalid input. Please enter numeric values only.")
        return

    # run the model to predict win/loss based on the individual performance
    input_df = pd.DataFrame([[kills, deaths, assists, total_gold, minions_killed]], columns=['kills', 'deaths', 'assists', 'TotalGold', 'MinionsKilled'])
    y_pred_individual = model_individual.predict(input_df)

    if y_pred_individual[0] == 1:
        print("The model predicts that you are likely to WIN this match based on your individual performance.")
    else:
        print("The model predicts that you are likely to LOSE this match based on your individual performance.")    

def team_performance(model_team):
    """
    Predicts the result of the team performance based on a user inputs of match stats, such as baron kills, rift herald kills, dragon kills, tower kills, kills, and average rank of the match.
    
    Args: 
        model_team: trained logistic regression model for team performance prediction.
    
    Returns:
        prints the predicted outcome (win/loss) of the team performance.
    """
        
    print("Enter your team performance stats for the match:")

    # parse the features, if invalid, clarify and ask again
    try:
        baron_kills = int(input("Baron Kills: ").strip())
        rift_herald_kills = int(input("Rift Herald Kills: ").strip())
        dragon_kills = int(input("Dragon Kills: ").strip())
        tower_kills = int(input("Tower Kills: ").strip())
        kills = int(input("Kills: ").strip())
    except ValueError:
        print("Invalid input. Please enter numeric values only.")
        return

    rank_name = input("Average Rank of match (e.g. Iron): ").strip()
    # convert rank name to rank id
    rank_data = pd.read_csv('kaggle/RankTbl.csv')
    rank_row = rank_data[rank_data['RankName'].str.lower() == rank_name.lower()]
    if rank_row.empty:
        print("Rank not found. Please check the rank name and try again.")
        return
    rank_id = rank_row['RankId'].values[0]

    # get the prediction for the team performance
    input_df = pd.DataFrame([[baron_kills, rift_herald_kills, dragon_kills, tower_kills, kills, rank_id]], columns=['BaronKills', 'RiftHeraldKills', 'DragonKills', 'TowerKills', 'Kills', 'RankFk'])
    y_pred_team = model_team.predict(input_df)

    if y_pred_team[0] == 1:
        print("The model predicts that your team is likely to WIN this match based on your team performance.")
    else:
        print("The model predicts that your team is likely to LOSE this match based on your team performance.")


def process_user_input(user_input, model_team, model_individual, model_lane):
    """
    Processes the user input, given the trained models for each type and the exact user input.
    
    Args: 
        user_input: the type of prediction the user wants to make (lane, individual, or team).
        model_team: trained logistic regression model for team performance prediction.
        model_individual: trained logistic regression model for individual performance prediction.
        model_lane: trained logistic regression model for lane matchup prediction.
    
    Returns:
        calls appropriate functions to get predictions.
    """
    if user_input == 'lane':
        lane_matchup(model_lane)
    elif user_input == 'individual':
        ind_performance(model_individual)
    else:
        team_performance(model_team)

def main():
    """
    Main function to run the prediction program.
    """
    model_team, model_individual, model_lane = preprocess_and_train()
    print("Type 'lane' to predict lane matchup outcomes\nType 'individual' to predict win/loss based on individual performance\nType 'team' to predict win/loss based on team performance\nType 'exit' to quit the program")

    while True:
        user_input = input("Enter your choice: ").strip().lower()
        if user_input in ['lane', 'individual', 'team']:
            process_user_input(user_input, model_team, model_individual, model_lane)
        elif user_input == 'exit':
            print("Exiting...")
            break
        else:
            print("Invalid input. Please try again.")

if __name__ == "__main__":
    main()
