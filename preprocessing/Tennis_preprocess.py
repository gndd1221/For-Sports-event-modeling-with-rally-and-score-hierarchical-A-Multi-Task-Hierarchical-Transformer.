import argparse
import pandas as pd
import numpy as np
import json
import os
import pickle
from sklearn.model_selection import train_test_split


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TENNIS_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'tennis')

def create_processed_data(config):
    """
    讀取原始 CSV，進行預處理，使其符合 5 層階層式結構 (Match->Set->Game->Rally->Shot)，
    並將資料分割為訓練、驗證和測試集。最後將處理過的資料和設定檔儲存起來。
    """
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 從 config 讀取資料分割參數 ---
    test_ratio = config.get('test_ratio', 0.2)
    val_ratio = config.get('val_ratio', 0.15) 
    random_seed = config.get('random_seed', 42)

    matches_orig = pd.read_csv(config['filename'])
    matches = matches_orig.copy()

    # --- 資料清理: 確保分數欄位為整數 ---
    score_cols = ['roundscore_A', 'roundscore_B', 'gamescore_A', 'gamescore_B']
    for col in score_cols:
        if col in matches.columns:
            matches[col] = matches[col].fillna(0).astype(int)

    # --- 特徵工程與編碼 ---
    # 類別特徵 (需要 +1 以預留 0 給 Padding)
    categorical_features_plus_one = [
        'type', 'backhand', 'location', 'spin', 
        'is_second_serve', 'is_tiebreak'
    ]
    
    for col in categorical_features_plus_one:
        if col in matches.columns:
            matches[col] = matches[col].fillna(0).astype(int) + 1
            num_classes = matches[col].max() + 1
            config[f'num_{col}'] = int(num_classes)
        else:
            print(f"Warning: Column '{col}' not found in dataset.")

    codes_player, uniques_player = pd.factorize(matches['player'])
    matches['player_id'] = codes_player + 1
    config['num_players'] = len(uniques_player) + 1
    player_map = {name: i + 1 for i, name in enumerate(uniques_player)}
    
    # 建立 rally 與 player 的對應關係
    rally_players_map = matches.groupby('rally_id')['player'].unique().apply(list).to_dict()
    
    def get_opponent(row):
        players = rally_players_map.get(row['rally_id'])
        if players and len(players) == 2:
            return players[1] if players[0] == row['player'] else players[0]
        return np.nan
        
    matches['opponent'] = matches.apply(get_opponent, axis=1)
    matches['opponent_id'] = matches['opponent'].map(player_map)
    matches.dropna(subset=['opponent_id'], inplace=True)
    matches['opponent_id'] = matches['opponent_id'].astype(int)
    
    # --- 建立階層式資料結構 (Match -> Set -> Game -> Rally -> Shot) ---
    features_to_extract = config['features_to_extract']
    all_matches_data = []
    
    # 確保資料依照順序排列
    matches = matches.sort_values(['match_id', 'set', 'rally_id', 'ball_round'])

    # Level 1: Match
    for match_id, match_df in matches.groupby('match_id', sort=False):
        match_data = {'match_id': match_id, 'sets': []}
        
        # Level 2: Set
        for set_num, set_df in match_df.groupby('set', sort=False):
            set_data = {'set': set_num, 'games': []}
            
            # Level 3: Game (根據 score_game 分組)
            for game_score, game_df in set_df.groupby('score_game', sort=False):
                game_data = {'score_game': game_score, 'rallies': []}
                
                # Level 4: Rally
                for rally_id, rally_df in game_df.groupby('rally_id', sort=False):
                    score_point = rally_df['score_point'].iloc[0]
                    
                    # Level 5: Shot
                    rally_shots = rally_df[features_to_extract].to_numpy()
                    
                    game_data['rallies'].append({
                        'rally_id': rally_id, 
                        'score_point': score_point,
                        'shots': rally_shots
                    })
                
                set_data['games'].append(game_data)
            match_data['sets'].append(set_data)
        all_matches_data.append(match_data)

    # --- 資料分割 ---
    match_ids = matches['match_id'].unique()
    
    # 1. Train / Temp
    train_val_test_ratio = val_ratio + test_ratio
    train_match_ids, temp_match_ids = train_test_split(
        match_ids, test_size=train_val_test_ratio, random_state=random_seed
    )
    
    # 2. Val / Test
    test_split_ratio = test_ratio / train_val_test_ratio
    val_match_ids, test_match_ids = train_test_split(
        temp_match_ids, test_size=test_split_ratio, random_state=random_seed
    )
    
    config['test_match_ids'] = [int(i) for i in test_match_ids]
    config['val_match_ids'] = [int(i) for i in val_match_ids]

    train_data = [m for m in all_matches_data if m['match_id'] in train_match_ids]
    test_data = [m for m in all_matches_data if m['match_id'] in test_match_ids]
    val_data = [m for m in all_matches_data if m['match_id'] in val_match_ids]

    # --- 儲存 ---
    with open(os.path.join(output_dir, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(output_dir, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    with open(os.path.join(output_dir, 'val_data.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
        
    with open(os.path.join(output_dir, 'player_map.json'), 'w', encoding='utf-8') as f:
        json.dump(player_map, f, indent=4, ensure_ascii=False)
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"資料處理完成。處理後的檔案已儲存於 '{output_dir}'")
    
    # --- 統計數據 ---
    train_ratio = 1.0 - val_ratio - test_ratio
    print(f"資料分割比例: 訓練集 {train_ratio:.0%}, 驗證集 {val_ratio:.0%}, 測試集 {test_ratio:.0%}")
    print(f"訓練集比賽數: {len(train_match_ids)}")
    print(f"驗證集比賽數: {len(val_match_ids)}")
    print(f"測試集比賽數: {len(test_match_ids)}")
    
    def count_stats(data):
        total_games = sum(len(s['games']) for m in data for s in m['sets'])
        total_rallies = sum(len(g['rallies']) for m in data for s in m['sets'] for g in s['games'])
        return total_games, total_rallies

    train_games, train_rallies = count_stats(train_data)
    val_games, val_rallies = count_stats(val_data)
    test_games, test_rallies = count_stats(test_data)

    print(f"訓練集 - 總局數: {train_games}, 總分(Rallies): {train_rallies}")
    print(f"驗證集 - 總局數: {val_games}, 總分(Rallies): {val_rallies}")
    print(f"測試集 - 總局數: {test_games}, 總分(Rallies): {test_rallies}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess tennis CSV data with match-level splits."
    )
    parser.add_argument('--variant', choices=['top100', 'top200'], default='top100')
    parser.add_argument('--input', dest='input_path', default=None, help='Override input CSV path.')
    parser.add_argument('--output-dir', default=None, help='Override processed-data directory.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    return parser.parse_args()


def build_config(args):
    if args.val_ratio <= 0 or args.test_ratio <= 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError('--val-ratio and --test-ratio must be positive and sum to less than 1.')
    variants = {
        'top100': ('Tennis_Converted_Top100.csv', 'Tennis_processed_data'),
        'top200': ('Tennis_Converted_Top200.csv', 'Tennis_processed_data_top200'),
    }
    filename, output_name = variants[args.variant]
    return {
        'variant': args.variant,
        'filename': args.input_path or os.path.join(TENNIS_DATA_DIR, filename),
        'output_dir': args.output_dir or os.path.join(TENNIS_DATA_DIR, output_name),
        'train_ratio': 1.0 - args.val_ratio - args.test_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'random_seed': args.seed,
        'features_to_extract': [
            'roundscore_A', 
            'roundscore_B',
            'gamescore_A',
            'gamescore_B',
            'player_id', 
            'opponent_id',
            'type', 
            'backhand', 
            'location', 
            'spin',
            'is_second_serve',
            'is_tiebreak',
            'set'
        ],
        'max_shot_seq_len': 35,
        'max_rally_seq_len': 25,
        'max_game_seq_len': 13,
        'max_set_seq_len': 5,
    }


if __name__ == '__main__':
    create_processed_data(build_config(parse_args()))
