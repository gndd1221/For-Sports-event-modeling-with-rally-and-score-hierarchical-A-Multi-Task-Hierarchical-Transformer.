import argparse
import pandas as pd
import numpy as np
import json
import os
import pickle
from sklearn.model_selection import train_test_split


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
TABLE_TENNIS_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'table_tennis')

def create_processed_data(config):
    """
    讀取原始 CSV，建立 MT-HTA 使用的階層式資料結構，
    並將資料分割為訓練、驗證和測試集。最後將處理過的資料和設定檔儲存起來。
    """
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    test_ratio = config.get('test_ratio', 0.2)
    val_ratio = config.get('val_ratio', 0.1)
    random_seed = config.get('random_seed', 42)

    matches_orig = pd.read_csv(config['filename'])
    matches = matches_orig.copy()

    required_columns = {
        'ball_round', 'roundscore_A', 'roundscore_B', 'player',
        'type', 'backhand', 'location', 'strength', 'spin',
        'set', 'match_id', 'rally_id'
    }
    missing_columns = sorted(required_columns - set(matches.columns))
    if missing_columns:
        raise ValueError(f"Missing required table tennis columns: {missing_columns}")

    # --- 特徵工程與編碼 ---
    categorical_features_plus_one = ['type', 'backhand', 'location', 'strength', 'spin']
    for col in categorical_features_plus_one:
        matches[col] = matches[col].astype(int) + 1
        num_classes = matches[col].max() + 1
        config[f'num_{col}'] = int(num_classes)
    codes_player, uniques_player = pd.factorize(matches['player'])
    matches['player_id'] = codes_player + 1
    config['num_players'] = len(uniques_player) + 1
    player_map = {name: i + 1 for i, name in enumerate(uniques_player)}
    rally_players_map = matches.groupby('rally_id')['player'].unique().apply(list).to_dict()
    invalid_rally_count = sum(len(players) != 2 for players in rally_players_map.values())

    def get_opponent(row):
        players = rally_players_map.get(row['rally_id'])
        if players and len(players) == 2:
            return players[1] if players[0] == row['player'] else players[0]
        return np.nan

    rows_before_opponent_drop = len(matches)
    matches_before_opponent_drop = matches['match_id'].nunique()
    matches['opponent'] = matches.apply(get_opponent, axis=1)
    matches['opponent_id'] = matches['opponent'].map(player_map)
    matches.dropna(subset=['opponent_id'], inplace=True)
    matches['opponent_id'] = matches['opponent_id'].astype(int)
    dropped_rows = rows_before_opponent_drop - len(matches)
    dropped_matches = matches_before_opponent_drop - matches['match_id'].nunique()
    
    # --- 建立階層式資料結構 ---
    features_to_extract = config['features_to_extract']
    all_matches_data = []
    for match_id, match_df in matches.groupby('match_id'):
        match_data = {'match_id': match_id, 'sets': []}
        for set_num, set_df in match_df.groupby('set'):
            set_data = {'set': set_num, 'rallies': []}
            for rally_id, rally_df in set_df.groupby('rally_id'):
                rally_df = rally_df.sort_values('ball_round').copy()
                rally_shots = rally_df[features_to_extract].to_numpy()
                set_data['rallies'].append({'rally_id': rally_id, 'shots': rally_shots})
            set_data['rallies'] = sorted(set_data['rallies'], key=lambda x: x['rally_id'])
            match_data['sets'].append(set_data)
        all_matches_data.append(match_data)

    # --- 以 match 為單位分割 train/validation/test ---
    match_ids = matches['match_id'].unique()
    
    # 1. 先分出訓練集 (train) 和 臨時集 (temp = val + test)
    train_val_test_ratio = val_ratio + test_ratio
    train_match_ids, temp_match_ids = train_test_split(
        match_ids, test_size=train_val_test_ratio, random_state=random_seed
    )
    
    # 2. 再從 臨時集 (temp) 中分出 驗證集 (val) 和 測試集 (test)
    #    計算測試集在臨時集中的比例
    test_split_ratio = test_ratio / train_val_test_ratio
    val_match_ids, test_match_ids = train_test_split(
        temp_match_ids, test_size=test_split_ratio, random_state=random_seed
    )
    
    config['test_match_ids'] = [int(i) for i in test_match_ids]
    config['val_match_ids'] = [int(i) for i in val_match_ids]

    train_data = [m for m in all_matches_data if m['match_id'] in train_match_ids]
    test_data = [m for m in all_matches_data if m['match_id'] in test_match_ids]
    val_data = [m for m in all_matches_data if m['match_id'] in val_match_ids]

    # --- 儲存處理後的資料 ---
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
    
    train_ratio = 1.0 - val_ratio - test_ratio
    print(f"資料分割比例: 訓練集 {train_ratio:.0%}, 驗證集 {val_ratio:.0%}, 測試集 {test_ratio:.0%}")
    print(f"訓練集比賽數: {len(train_match_ids)}")
    print(f"驗證集比賽數: {len(val_match_ids)}")
    print(f"測試集比賽數: {len(test_match_ids)}")
    if dropped_rows:
        print(
            "移除無法建立 opponent 的資料: "
            f"{invalid_rally_count} 個 rally, {dropped_rows} 筆擊球事件, {dropped_matches} 場比賽"
        )
    print("-" * 30)
    print(f"訓練集總回合數: {sum(len(s['rallies']) for m in train_data for s in m['sets'])}")
    print(f"驗證集總回合數: {sum(len(s['rallies']) for m in val_data for s in m['sets'])}")
    print(f"測試集總回合數: {sum(len(s['rallies']) for m in test_data for s in m['sets'])}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess table-tennis CSV data with match-level splits."
    )
    parser.add_argument('--variant', choices=['original', 'all'], default='original')
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
        'original': ('TableTennis_dataset.csv', 'processed_data'),
        'all': ('TableTennis_dataset_all.csv', 'processed_data_all'),
    }
    filename, output_name = variants[args.variant]
    return {
        'variant': args.variant,
        'filename': args.input_path or os.path.join(TABLE_TENNIS_DATA_DIR, filename),
        'output_dir': args.output_dir or os.path.join(TABLE_TENNIS_DATA_DIR, output_name),
        'train_ratio': 1.0 - args.val_ratio - args.test_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'random_seed': args.seed,
        'features_to_extract': [
            'roundscore_A', 
            'roundscore_B',
            'player_id', 
            'opponent_id',
            'type', 
            'backhand', 
            'location', 
            'strength', 
            'spin',
            'set'
        ],
        'max_shot_seq_len': 35,
        'max_rally_seq_len': 30,
        'max_set_seq_len': 7,
    }


if __name__ == '__main__':
    create_processed_data(build_config(parse_args()))
