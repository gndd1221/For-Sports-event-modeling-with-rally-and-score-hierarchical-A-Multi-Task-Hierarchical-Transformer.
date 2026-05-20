import os
import sys
import pickle
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 桌球落點 1-9 的網格對應 (3x3 矩陣)
# 1:正短, 2:中短, 3:反短
# 4:正半, 5:中半, 6:反半
# 7:正長, 8:中長, 9:反長
LOCATION_GRID = {
    1: (0, 2), 2: (1, 2), 3: (2, 2),
    4: (0, 1), 5: (1, 1), 6: (2, 1),
    7: (0, 0), 8: (1, 0), 9: (2, 0)
}
# X軸: 正手(0) -> 中間(1) -> 反手(2)
# Y軸: 長球(0) -> 半出台(1) -> 短球(2) (靠網)

SHOT_TYPE_MAP = {
    0: "Zero", 1: "Drive", 2: "Counter", 3: "Smash", 4: "Twist",
    5: "FastDrive", 6: "FastPush(A)", 7: "Flip", 8: "LongPush(P)", 
    9: "FastPush(P)", 10: "LongPush", 11: "Drop", 12: "Chop", 
    13: "Block", 14: "Lob", 15: "Serve(Trad)", 16: "Serve(Hook)", 
    17: "Serve(Rev)", 18: "Serve(Squat)"
}

def load_data(sport, data_split='train'):
    # Adjust data_dir based on standard sport structures in this project
    if sport == 'badminton':
        data_dir = os.path.join(_project_root, 'data', sport, 'processed_data_badminton')
    elif sport == 'tennis':
        data_dir = os.path.join(_project_root, 'data', sport, 'Tennis_processed_data')
    else:
        # Default for table_tennis
        data_dir = os.path.join(_project_root, 'data', sport, 'processed_data')
        
    data_path = os.path.join(data_dir, f'{data_split}_data.pkl')
    config_path = os.path.join(data_dir, 'config.json')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find {data_path}")
        
    with open(data_path, 'rb') as f:
        matches = pickle.load(f)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    features = config['features_to_extract']
    feat_idx = {name: i for i, name in enumerate(features)}
    
    return matches, feat_idx

def parse_rallies(matches, feat_idx):
    all_shots = []
    transitions_global = []
    transitions_1_2 = []
    transitions_2_3 = []
    
    type_idx = feat_idx.get('type')
    loc_idx = feat_idx.get('location')
    spin_idx = feat_idx.get('spin')
    str_idx = feat_idx.get('strength')
    
    if type_idx is None or loc_idx is None:
        raise ValueError("Config must have 'type' and 'location' in features_to_extract")

    for match in matches:
        for set_data in match.get('sets', []):
            # Support 3L (rallies directly in sets) or 4L (games then rallies)
            rallies = []
            if 'games' in set_data:
                for game in set_data['games']:
                    rallies.extend(game.get('rallies', []))
            else:
                rallies = set_data.get('rallies', [])
                
            for rally in rallies:
                shots = rally.get('shots', [])
                if len(shots) == 0:
                    continue
                
                # Extract shot features
                for i, shot in enumerate(shots):
                    # In preprocessing, all categorical features were shifted by +1 to reserve 0 for padding.
                    # We subtract 1 here to map back to the original CSV semantics.
                    stype = int(shot[type_idx]) - 1
                    sloc = int(shot[loc_idx]) - 1
                    sspin = int(shot[spin_idx]) - 1 if spin_idx is not None else 0
                    sstr = int(shot[str_idx]) - 1 if str_idx is not None else 0
                    
                    all_shots.append({
                        'rally_id': id(rally),
                        'shot_num': i + 1,
                        'type': stype,
                        'location': sloc,
                        'spin': sspin,
                        'strength': sstr
                    })
                    
                    # Transitions
                    if i > 0:
                        prev_type = int(shots[i-1][type_idx]) - 1
                        transitions_global.append((prev_type, stype))
                        
                        if i == 1:
                            transitions_1_2.append((prev_type, stype))
                        elif i == 2:
                            transitions_2_3.append((prev_type, stype))
                            
    df_shots = pd.DataFrame(all_shots)
    return df_shots, transitions_global, transitions_1_2, transitions_2_3

def plot_transition_matrix(transitions, num_types, title, filename, out_dir):
    matrix = np.zeros((num_types, num_types))
    for prev_t, next_t in transitions:
        if prev_t < num_types and next_t < num_types:
            matrix[prev_t, next_t] += 1
            
    # 將 0 (Zero) 納入機率計算
    active_types = [i for i in range(0, 19)]  # 0 to 18
    sub_matrix = matrix[0:19, 0:19]
    
    # 對提取出來的有效矩陣做 Row-Normalization (保證看見的格子加起來是 1.0)
    row_sums = sub_matrix.sum(axis=1, keepdims=True)
    sub_matrix_prob = np.divide(sub_matrix, row_sums, out=np.zeros_like(sub_matrix), where=row_sums!=0)
    
    labels = [SHOT_TYPE_MAP.get(i, str(i)) for i in active_types]
    
    plt.figure(figsize=(15, 13))
    sns.heatmap(sub_matrix_prob, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 8})
    plt.title(title, fontsize=16)
    plt.xlabel('Next Shot Type (t)', fontsize=12)
    plt.ylabel('Previous Shot Type (t-1)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()

def plot_spatial_heatmap(df_shots, out_dir):
    # 準備要畫的資料清單
    plot_data = []
    
    # 1. 所有球種 (All Shots)
    plot_data.append(("All Shots Combined", df_shots))
    
    # 2. 所有發球 (All Serves)
    serve_df = df_shots[(df_shots['type'] >= 15) & (df_shots['type'] <= 18)]
    plot_data.append(("All Serves (Types 15-18)", serve_df))
    
    # 3. 前 4 大一般球種 (Top 4 Regular Shots, 包含 0~14)
    play_shots = df_shots[(df_shots['type'] >= 0) & (df_shots['type'] <= 14)]
    top_types = play_shots['type'].value_counts().nlargest(4).index.tolist()
    
    for stype in top_types:
        type_name = SHOT_TYPE_MAP.get(stype, f"Type-{stype}")
        type_df = df_shots[df_shots['type'] == stype]
        plot_data.append((type_name, type_df))
        
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, (title, df_subset) in enumerate(plot_data):
        # 3x3 Grid
        grid = np.zeros((3, 3))
        for loc, count in df_subset['location'].value_counts().items():
            if loc in LOCATION_GRID:
                x, y = LOCATION_GRID[loc]
                grid[y, x] += count  # y is row, x is col
                
        # Normalize
        if grid.sum() > 0:
            grid = grid / grid.sum()
            
        sns.heatmap(grid, annot=True, fmt='.1%', cmap='Reds', ax=axes[i], 
                    xticklabels=['Forehand', 'Middle', 'Backhand'],
                    yticklabels=['Long', 'Half-Long', 'Short (Net)'])
        axes[i].set_title(f'Spatial Distribution: {title}\n(n={len(df_subset)})')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'spatial_kde_comprehensive.png'), dpi=300)
    plt.close()

def plot_correlation(df_shots, out_dir):
    # Select features for correlation
    cols = ['type', 'location', 'spin', 'strength']
    avail_cols = [c for c in cols if c in df_shots.columns]
    
    # 將 0 也視為一種狀態，不進行過濾 (直接使用所有 shots)
    valid_shots = df_shots
    
    # Using Spearman because these are largely ordinal/categorical indices
    corr = valid_shots[avail_cols].corr(method='spearman')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Spearman Correlation between Multi-Task Features")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_correlation.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sport', type=str, default='table_tennis')
    parser.add_argument('--split', type=str, default='train', help='train, val, or test')
    parser.add_argument('--out_dir', type=str, default='outputs/eda')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading {args.split} data for {args.sport}...")
    
    matches, feat_idx = load_data(args.sport, args.split)
    df_shots, trans_global, trans_1_2, trans_2_3 = parse_rallies(matches, feat_idx)
    
    print(f"Total shots parsed: {len(df_shots)}")
    
    print("1. Generating Shot Transition Matrices...")
    plot_transition_matrix(trans_global, 20, 
                           "Global Shot Transition Matrix (t-1 -> t)", 
                           "transition_matrix_global.png", args.out_dir)
    plot_transition_matrix(trans_1_2, 20, 
                           "Serve & Return Transition (Shot 1 -> Shot 2)", 
                           "transition_matrix_first_1_2.png", args.out_dir)
    plot_transition_matrix(trans_2_3, 20, 
                           "Third-Ball Attack Transition (Shot 2 -> Shot 3)", 
                           "transition_matrix_first_2_3.png", args.out_dir)
                           
    print("2. Generating Spatial KDE Heatmaps...")
    plot_spatial_heatmap(df_shots, args.out_dir)
    
    print("3. Generating Feature Correlation Matrix...")
    plot_correlation(df_shots, args.out_dir)
    
    print(f"All EDA plots saved to {args.out_dir}/")

if __name__ == '__main__':
    main()
