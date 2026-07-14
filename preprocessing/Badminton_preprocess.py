import argparse
import pandas as pd
import numpy as np
import json
import os
import pickle
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BADMINTON_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'badminton')

def create_processed_data(config):
    """
    讀取羽球 CSV，建立 MT-HTA 使用的階層式資料結構，
    並將資料分割為訓練、驗證和測試集。最後將處理過的資料和設定檔儲存起來。
    """
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 從 config 讀取資料分割參數 ---
    test_ratio = config.get('test_ratio', 0.2)
    val_ratio = config.get('val_ratio', 0.1) 
    random_seed = config.get('random_seed', 42)

    print(f"正在讀取資料: {config['filename']}...")
    matches_orig = pd.read_csv(config['filename'])
    matches = matches_orig.copy()

    # --- 特徵工程與編碼 ---
    
    # 1. 處理 Type (球種): 因為是文字，需要 Factorize
    matches['type'] = matches['type'].astype(str)
    codes_type, uniques_type = pd.factorize(matches['type'])
    matches['type'] = codes_type + 1 # 0 保留給 padding
    config['num_type'] = len(uniques_type) + 1
    print(f"球種 (Type) 類別數: {len(uniques_type)}")

    # 2. 處理數值型類別特徵 (Backhand, Areas)
    # 注意：原始資料中區域可能是 1~N，我們全部 +1 以便讓 0 做為 Padding
    # 同時處理可能的 NaN 值 (填 0，這樣 +1 後變成 1，代表未知或特定區域，或視需求調整)
    cat_features = ['backhand', 'landing_area', 'player_location_area', 'opponent_location_area']
    
    for col in cat_features:
        # 填補缺失值為 -1 (之後 +1 變 0) 或其他預設值
        matches[col] = matches[col].fillna(0).astype(int) 
        # 將數值 +1，確保從 1 開始 (0 保留給 Padding)
        # 如果原始資料已經有 0，這一步會讓它變成 1
        matches[col] = matches[col] + 1
        num_classes = matches[col].max() + 1
        config[f'num_{col}'] = int(num_classes)
        print(f"特徵 {col} 類別數: {num_classes}")

    # 3. 處理球員 ID
    codes_player, uniques_player = pd.factorize(matches['player'])
    matches['player_id'] = codes_player + 1
    config['num_players'] = len(uniques_player) + 1
    
    player_map = {name: i + 1 for i, name in enumerate(uniques_player)}
    
    # 找出對手 ID
    rally_players_map = matches.groupby('rally_id')['player'].unique().apply(list).to_dict()
    
    def get_opponent(row):
        players = rally_players_map.get(row['rally_id'])
        if players and len(players) >= 1:
            # 如果 rally 只有一個人(極少見)，或者正常兩人
            for p in players:
                if p != row['player']:
                    return p
            # 如果找不到對手 (單人練球紀錄?), 回傳自己或 NaN
            return np.nan 
        return np.nan

    matches['opponent'] = matches.apply(get_opponent, axis=1)
    # 移除找不到對手的資料
    matches.dropna(subset=['opponent'], inplace=True)
    
    matches['opponent_id'] = matches['opponent'].map(player_map)
    # 若對手不在 player 列表中 (罕見)，填 0 或 drop
    matches.dropna(subset=['opponent_id'], inplace=True)
    matches['opponent_id'] = matches['opponent_id'].astype(int)
    
    # --- 建立階層式資料結構 ---
    print("正在建立階層式資料結構...")
    features_to_extract = config['features_to_extract']
    all_matches_data = []
    
    # 根據 match_id 分組
    for match_id, match_df in matches.groupby('match_id'):
        match_data = {'match_id': match_id, 'sets': []}
        # 根據 set 分組
        for set_num, set_df in match_df.groupby('set'):
            set_data = {'set': set_num, 'rallies': []}
            # 根據 rally_id 分組
            for rally_id, rally_df in set_df.groupby('rally_id'):
                # 確保按照擊球順序 (ball_round) 排序
                rally_df = rally_df.sort_values('ball_round').copy()
                rally_shots = rally_df[features_to_extract].to_numpy()
                set_data['rallies'].append({'rally_id': rally_id, 'shots': rally_shots})
            
            # 確保 Rally 順序正確 (通常 rally_id 是遞增的)
            set_data['rallies'] = sorted(set_data['rallies'], key=lambda x: x['rally_id'])
            match_data['sets'].append(set_data)
        all_matches_data.append(match_data)

    # --- 資料分割 ---
    match_ids = matches['match_id'].unique()
    
    # 1. Train / Temp Split
    train_val_test_ratio = val_ratio + test_ratio
    train_match_ids, temp_match_ids = train_test_split(
        match_ids, test_size=train_val_test_ratio, random_state=random_seed
    )
    
    # 2. Val / Test Split
    if len(temp_match_ids) > 0:
        test_split_ratio = test_ratio / train_val_test_ratio
        # 避免除以零或錯誤
        if test_split_ratio >= 1.0: 
            val_match_ids = []
            test_match_ids = temp_match_ids
        else:
            val_match_ids, test_match_ids = train_test_split(
                temp_match_ids, test_size=test_split_ratio, random_state=random_seed
            )
    else:
        val_match_ids = []
        test_match_ids = []
    
    config['test_match_ids'] = [int(i) for i in test_match_ids]
    config['val_match_ids'] = [int(i) for i in val_match_ids]

    train_data = [m for m in all_matches_data if m['match_id'] in train_match_ids]
    val_data = [m for m in all_matches_data if m['match_id'] in val_match_ids]
    test_data = [m for m in all_matches_data if m['match_id'] in test_match_ids]

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
    print(f"資料分割: Train {len(train_data)} / Val {len(val_data)} / Test {len(test_data)} (Matches)")
    print(f"總回合數 (Rallies): Train {sum(len(s['rallies']) for m in train_data for s in m['sets'])}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess badminton CSV data with match-level splits."
    )
    parser.add_argument('--variant', choices=['original', 'all'], default='all')
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
        'original': ('badminton_dataset.csv', 'processed_data_badminton'),
        'all': ('badminton_dataset_all.csv', 'processed_data_badminton_all'),
    }
    filename, output_name = variants[args.variant]
    return {
        'variant': args.variant,
        'filename': args.input_path or os.path.join(BADMINTON_DATA_DIR, filename),
        'output_dir': args.output_dir or os.path.join(BADMINTON_DATA_DIR, output_name),
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
            'landing_area',
            'player_location_area',
            'opponent_location_area',
            'set'
        ],
        'categorical_features': [
            'type', 'backhand', 'landing_area', 
            'player_location_area', 'opponent_location_area'
        ],
        'numerical_features': ['roundscore_A', 'roundscore_B', 'set'],
        
        'max_shot_seq_len': 55,
        'max_rally_seq_len': 45,
        'max_set_seq_len': 3,
    }


if __name__ == '__main__':
    create_processed_data(build_config(parse_args()))
