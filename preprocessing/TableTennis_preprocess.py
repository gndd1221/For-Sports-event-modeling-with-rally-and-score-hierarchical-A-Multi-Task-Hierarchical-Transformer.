import pandas as pd
import numpy as np
import json
import os
import pickle
from sklearn.model_selection import train_test_split

def create_processed_data(config):
    """
    讀取原始 CSV，進行預處理，使其符合 PACT 模型所需的階層式結構，
    並將資料分割為訓練、驗證和測試集。最後將處理過的資料和設定檔儲存起來。
    """
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 從 config 讀取資料分割參數 ---
    test_ratio = config.get('test_ratio', 0.2)
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    #  新增: 讀取驗證集比例
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    val_ratio = config.get('val_ratio', 0.15) 
    random_seed = config.get('random_seed', 42)

    matches_orig = pd.read_csv(config['filename'])
    matches = matches_orig.copy()

    # --- 特徵工程與編碼 (此部分不變) ---
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
    def get_opponent(row):
        players = rally_players_map.get(row['rally_id'])
        if players and len(players) == 2:
            return players[1] if players[0] == row['player'] else players[0]
        return np.nan
    matches['opponent'] = matches.apply(get_opponent, axis=1)
    matches['opponent_id'] = matches['opponent'].map(player_map)
    matches.dropna(subset=['opponent_id'], inplace=True)
    matches['opponent_id'] = matches['opponent_id'].astype(int)
    
    # --- 建立階層式資料結構 (此部分不變) ---
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

    # --- 資料分割 (修改為三份) ---
    match_ids = matches['match_id'].unique()
    
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    #  修改: 進行兩次分割，產生 train, val, test
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    
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
    config['val_match_ids'] = [int(i) for i in val_match_ids] # 新增

    train_data = [m for m in all_matches_data if m['match_id'] in train_match_ids]
    test_data = [m for m in all_matches_data if m['match_id'] in test_match_ids]
    val_data = [m for m in all_matches_data if m['match_id'] in val_match_ids] # 新增

    # --- 儲存處理後的資料 ---
    with open(os.path.join(output_dir, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    with open(os.path.join(output_dir, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    #  新增: 儲存驗證集 .pkl
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    with open(os.path.join(output_dir, 'val_data.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
        
    with open(os.path.join(output_dir, 'player_map.json'), 'w', encoding='utf-8') as f:
        json.dump(player_map, f, indent=4, ensure_ascii=False)
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"資料處理完成。處理後的檔案已儲存於 '{output_dir}'")
    
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    #  修改: 更新日誌輸出
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    train_ratio = 1.0 - val_ratio - test_ratio
    print(f"資料分割比例: 訓練集 {train_ratio:.0%}, 驗證集 {val_ratio:.0%}, 測試集 {test_ratio:.0%}")
    print(f"訓練集比賽數: {len(train_match_ids)}")
    print(f"驗證集比賽數: {len(val_match_ids)}")
    print(f"測試集比賽數: {len(test_match_ids)}")
    print("-" * 30)
    print(f"訓練集總回合數: {sum(len(s['rallies']) for m in train_data for s in m['sets'])}")
    print(f"驗證集總回合數: {sum(len(s['rallies']) for m in val_data for s in m['sets'])}")
    print(f"測試集總回合數: {sum(len(s['rallies']) for m in test_data for s in m['sets'])}")

if __name__ == '__main__':
    config = {
        'filename': 'TableTennis_dataset.csv',
        'output_dir': 'processed_data',
        
        # --- 資料分割設定 ---
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        'train_ratio': 0.7, # 修改 (1.0 - 0.2 - 0.15)
        'val_ratio': 0.1,   # 新增
        'test_ratio': 0.2, 
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        'random_seed': 42,
        
        # --- 模型特徵與長度設定 ---
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
        'max_shot_seq_len': 50,
        'max_rally_seq_len': 32,
        'max_set_seq_len': 5,
    }
    create_processed_data(config)