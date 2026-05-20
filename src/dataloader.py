"""
dataloader.py - 統一所有運動的資料載入模組

支援 3 層 (桌球/羽球: Shot→Rally→Set) 和 4 層 (網球: Shot→Rally→Game→Set) 架構。
透過 get_dataloader() 工廠函式自動選擇對應的 Dataset + collate_fn。
"""
import torch
import numpy as np
import pickle
import json
from torch.utils.data import Dataset, DataLoader, Sampler


# =====================================================================================
# 1. PACTDataset — 3 層架構 (桌球/羽球: Match→Set→Rally→Shot)
# =====================================================================================
class PACTDataset(Dataset):
    """
    適用於 3 層階層式模型 (Shot→Rally→Set) 的 Dataset。
    targets 從 config['targets'] 動態讀取，不再 hardcode 任何運動專屬的 key。
    """
    def __init__(self, data_path, config_path, target_names=None, max_lens=None):
        super().__init__()
        with open(data_path, 'rb') as f:
            self.all_matches = pickle.load(f)
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        features = self.config['features_to_extract']
        self.feature_indices = {name: i for i, name in enumerate(features)}
        # target_names 優先使用傳入值，再從 config 讀取
        self.target_names = target_names or self.config.get('targets', list(self.config.get('loss_weights', {}).keys()))

        # 硬上限截斷設定：None 表示不截斷（向後相容）
        _ml = max_lens or {}
        self.max_shot_seq_len  = _ml.get('max_shot_seq_len',  None)
        self.max_rally_seq_len = _ml.get('max_rally_seq_len', None)
        self.max_game_seq_len  = _ml.get('max_game_seq_len',  None)
        self.max_set_seq_len   = _ml.get('max_set_seq_len',   None)

        self.samples = self._create_samples()

    def _create_samples(self):
        """遍歷 Match→Set→Rally→Shot，為每一拍建立樣本。"""
        samples = []
        for match_idx, match in enumerate(self.all_matches):
            for set_idx, set_data in enumerate(match['sets']):
                for rally_idx, rally_data in enumerate(set_data['rallies']):
                    shots = rally_data['shots']
                    if len(shots) >= 2:
                        for target_shot_idx in range(1, len(shots)):
                            samples.append({
                                'match_idx': match_idx,
                                'set_idx': set_idx,
                                'rally_idx': rally_idx,
                                'target_shot_idx': target_shot_idx
                            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        match_idx = sample_info['match_idx']
        set_idx = sample_info['set_idx']
        rally_idx = sample_info['rally_idx']
        target_shot_idx = sample_info['target_shot_idx']

        match_data = self.all_matches[match_idx]
        set_data = match_data['sets'][set_idx]
        rally_data = set_data['rallies'][rally_idx]

        # L1: 當前回合擊球序列（硬上限截斷：保留最近 N 拍）
        shot_seq_current = rally_data['shots'][:target_shot_idx]
        if self.max_shot_seq_len is not None:
            shot_seq_current = shot_seq_current[-self.max_shot_seq_len:]

        # L2: 歷史回合序列（硬上限截斷：保留最近 N 條 Rally）
        rally_history = set_data['rallies'][:rally_idx]
        if self.max_rally_seq_len is not None:
            rally_history = rally_history[-self.max_rally_seq_len:]

        # L3: 歷史局序列（硬上限截斷：保留最近 N 個 Set）
        set_history = match_data['sets'][:set_idx]
        if self.max_set_seq_len is not None:
            set_history = set_history[-self.max_set_seq_len:]

        # Targets (動態從 config 讀取)
        target_shot = rally_data['shots'][target_shot_idx]
        targets = {
            name: target_shot[self.feature_indices[name]]
            for name in self.target_names
        }

        # 球員 ID
        player_id = target_shot[self.feature_indices['player_id']]
        opponent_id = target_shot[self.feature_indices['opponent_id']]

        return {
            'shot_seq_current': np.array(shot_seq_current, dtype=np.int64),
            'rally_history': rally_history,
            'set_history': set_history,
            'player_id': player_id,
            'opponent_id': opponent_id,
            'targets': targets
        }


# =====================================================================================
# 2. PACTDataset4L — 4 層架構 (網球: Match→Set→Game→Rally→Shot)
# =====================================================================================
class PACTDataset4L(PACTDataset):
    """
    適用於 4 層階層式模型 (Shot→Rally→Game→Set) 的 Dataset。
    繼承 PACTDataset，覆寫 _create_samples 和 __getitem__ 以處理 Game 層。
    """

    def _create_samples(self):
        """遍歷 Match→Set→Game→Rally→Shot，為每一拍建立樣本。"""
        samples = []
        for match_idx, match in enumerate(self.all_matches):
            for set_idx, set_data in enumerate(match['sets']):
                for game_idx, game_data in enumerate(set_data['games']):
                    for rally_idx, rally_data in enumerate(game_data['rallies']):
                        shots = rally_data['shots']
                        if len(shots) >= 2:
                            for target_shot_idx in range(1, len(shots)):
                                samples.append({
                                    'match_idx': match_idx,
                                    'set_idx': set_idx,
                                    'game_idx': game_idx,
                                    'rally_idx': rally_idx,
                                    'target_shot_idx': target_shot_idx
                                })
        return samples

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        match_idx = sample_info['match_idx']
        set_idx = sample_info['set_idx']
        game_idx = sample_info['game_idx']
        rally_idx = sample_info['rally_idx']
        target_shot_idx = sample_info['target_shot_idx']

        match_data = self.all_matches[match_idx]
        set_data = match_data['sets'][set_idx]
        game_data = set_data['games'][game_idx]
        rally_data = game_data['rallies'][rally_idx]

        # L1: 當前 Shot 序列（硬上限截斷：保留最近 N 拍）
        shot_seq_current = rally_data['shots'][:target_shot_idx]
        if self.max_shot_seq_len is not None:
            shot_seq_current = shot_seq_current[-self.max_shot_seq_len:]

        # L2: 歷史 Rally 序列 (當前 Game 內)（硬上限截斷：保留最近 N 條 Rally）
        rally_history = game_data['rallies'][:rally_idx]
        if self.max_rally_seq_len is not None:
            rally_history = rally_history[-self.max_rally_seq_len:]

        # L3: 歷史 Game 序列 (當前 Set 內)（硬上限截斷：保留最近 N 個 Game）
        game_history = set_data['games'][:game_idx]
        if self.max_game_seq_len is not None:
            game_history = game_history[-self.max_game_seq_len:]

        # L4: 歷史 Set 序列 (當前 Match 內)（硬上限截斷：保留最近 N 個 Set）
        set_history = match_data['sets'][:set_idx]
        if self.max_set_seq_len is not None:
            set_history = set_history[-self.max_set_seq_len:]

        # Targets (動態從 config 讀取)
        target_shot = rally_data['shots'][target_shot_idx]
        targets = {
            name: target_shot[self.feature_indices[name]]
            for name in self.target_names
        }

        # 球員 ID
        player_id = target_shot[self.feature_indices['player_id']]
        opponent_id = target_shot[self.feature_indices['opponent_id']]

        return {
            'shot_seq_current': np.array(shot_seq_current, dtype=np.int64),
            'rally_history': rally_history,
            'game_history': game_history,
            'set_history': set_history,
            'player_id': player_id,
            'opponent_id': opponent_id,
            'targets': targets
        }


# =====================================================================================
# 3. Collate Function — 3 層版 (桌球/羽球)
# =====================================================================================
def _pad_nested_sequence(sequences, max_outer_len, num_features, max_inner_len=None):
    """輔助函數：填充巢狀序列的第一層和第二層"""
    if max_inner_len is None:
        max_inner_len = max(seq.shape[0] for seq in sequences) if sequences else 0

    padded_sequences = torch.zeros(max_outer_len, max_inner_len, num_features, dtype=torch.long)
    lengths = torch.zeros(max_outer_len, dtype=torch.long)

    for i, seq in enumerate(sequences):
        end = min(seq.shape[0], max_inner_len)
        padded_sequences[i, :end, :] = seq[:end]
        lengths[i] = end

    return padded_sequences, lengths


def collate_fn_3L(batch):
    """
    3 層 collate_fn (桌球/羽球)。
    處理 L1 (shot_seq_current), L2 (rally_history), L3 (set_history)。
    輸出 set_history 為 5D: (B, S, R, T, F)。
    """
    # --- L1: shot_seq_current ---
    shot_seqs = [torch.from_numpy(item['shot_seq_current']) for item in batch]
    shot_lengths = torch.tensor([len(seq) for seq in shot_seqs], dtype=torch.long)
    padded_shot_seqs = torch.nn.utils.rnn.pad_sequence(shot_seqs, batch_first=True, padding_value=0)
    num_features = padded_shot_seqs.shape[-1]

    rally_histories = [item['rally_history'] for item in batch]
    set_histories = [item['set_history'] for item in batch]

    # 計算所有巢狀序列中的最大 shot 長度
    _rally_shot_lens = [rally['shots'].shape[0] for hist in rally_histories for rally in hist]
    max_shots_in_rallies = max(_rally_shot_lens) if _rally_shot_lens else 0

    _set_shot_lens = [rally['shots'].shape[0] for hist in set_histories for set_data in hist for rally in set_data['rallies']]
    max_shots_in_sets = max(_set_shot_lens) if _set_shot_lens else 0
    max_shot_len_for_history = max(max_shots_in_rallies, max_shots_in_sets, 1)

    # --- L2: rally_history ---
    rally_history_lengths = torch.tensor([len(hist) for hist in rally_histories], dtype=torch.long)
    max_rallies_in_hist = rally_history_lengths.max().item() if len(rally_history_lengths) > 0 else 0

    padded_rally_history = []
    rally_shot_lengths = []

    for hist in rally_histories:
        rallies_in_hist = [torch.from_numpy(rally['shots']) for rally in hist]
        padded_rallies, lengths = _pad_nested_sequence(
            rallies_in_hist,
            max_outer_len=max_rallies_in_hist,
            max_inner_len=max_shot_len_for_history,
            num_features=num_features
        )
        padded_rally_history.append(padded_rallies)
        rally_shot_lengths.append(lengths)

    final_rally_history = torch.stack(padded_rally_history) if padded_rally_history else torch.empty(0)
    final_rally_shot_lengths = torch.stack(rally_shot_lengths) if rally_shot_lengths else torch.empty(0)

    # --- L3: set_history (5D: B, S, R, T, F) ---
    set_history_lengths = torch.tensor([len(hist) for hist in set_histories], dtype=torch.long)
    max_sets_in_hist = set_history_lengths.max().item() if len(set_history_lengths) > 0 else 0
    max_rallies_in_sets = max(
        [len(set_data['rallies']) for hist in set_histories for set_data in hist]
        if any(set_histories) else [0]
    )

    padded_set_history = []
    set_rally_lengths_list = []
    set_shot_lengths_list = []

    for hist in set_histories:
        padded_sets = []
        rally_lengths_for_sets = []
        shot_lengths_for_sets = []

        for set_data in hist:
            rallies_in_set = [torch.from_numpy(rally['shots']) for rally in set_data['rallies']]
            padded_rallies, lengths = _pad_nested_sequence(
                rallies_in_set,
                max_outer_len=max_rallies_in_sets,
                max_inner_len=max_shot_len_for_history,
                num_features=num_features
            )
            padded_sets.append(padded_rallies)
            rally_lengths_for_sets.append(len(rallies_in_set))
            shot_lengths_for_sets.append(lengths)

        num_current_sets = len(hist)
        if num_current_sets > 0:
            final_padded_sets = torch.stack(padded_sets)
            final_shot_lengths = torch.stack(shot_lengths_for_sets)
            final_rally_lengths = torch.tensor(rally_lengths_for_sets, dtype=torch.long)
        else:
            final_padded_sets = torch.zeros((0, max_rallies_in_sets, max_shot_len_for_history, num_features), dtype=torch.long)
            final_shot_lengths = torch.zeros((0, max_rallies_in_sets), dtype=torch.long)
            final_rally_lengths = torch.zeros((0,), dtype=torch.long)

        num_pad_sets = max_sets_in_hist - num_current_sets
        if num_pad_sets > 0:
            pad_shape_sets = (num_pad_sets, max_rallies_in_sets, max_shot_len_for_history, num_features)
            final_padded_sets = torch.cat([final_padded_sets, torch.zeros(pad_shape_sets, dtype=torch.long)], dim=0)

            pad_shape_shot_len = (num_pad_sets, max_rallies_in_sets)
            final_shot_lengths = torch.cat([final_shot_lengths, torch.zeros(pad_shape_shot_len, dtype=torch.long)], dim=0)

            pad_shape_rally_len = (num_pad_sets,)
            final_rally_lengths = torch.cat([final_rally_lengths, torch.zeros(pad_shape_rally_len, dtype=torch.long)], dim=0)

        padded_set_history.append(final_padded_sets)
        set_shot_lengths_list.append(final_shot_lengths)
        set_rally_lengths_list.append(final_rally_lengths)

    final_set_history = torch.stack(padded_set_history) if padded_set_history else torch.empty(0)
    final_set_shot_lengths = torch.stack(set_shot_lengths_list) if set_shot_lengths_list else torch.empty(0)
    final_set_rally_lengths = torch.stack(set_rally_lengths_list) if set_rally_lengths_list else torch.empty(0)

    # --- Targets & Meta ---
    player_ids = torch.tensor([item['player_id'] for item in batch], dtype=torch.long)
    opponent_ids = torch.tensor([item['opponent_id'] for item in batch], dtype=torch.long)
    targets = {key: torch.tensor([item['targets'][key] for item in batch], dtype=torch.long) for key in batch[0]['targets']}

    return {
        'shot_seq_current': padded_shot_seqs,
        'shot_seq_current_lengths': shot_lengths,
        'rally_history': final_rally_history,
        'rally_history_lengths': rally_history_lengths,
        'rally_history_shot_lengths': final_rally_shot_lengths,
        'set_history': final_set_history,
        'set_history_lengths': set_history_lengths,
        'set_history_rally_lengths': final_set_rally_lengths,
        'set_history_shot_lengths': final_set_shot_lengths,
        'player_id': player_ids,
        'opponent_id': opponent_ids,
        'targets': targets
    }


# =====================================================================================
# 4. Collate Function — 4 層版 (網球)
# =====================================================================================
def collate_fn_4L(batch):
    """
    4 層 collate_fn (網球)。
    處理 L1 (shot), L2 (rally), L3 (game_history), L4 (set_history)。
    輸出 game_history 為 5D: (B, G, R, T, F)
    輸出 set_history 為 6D: (B, St, G, R, T, F)
    """
    # --- L1: shot_seq_current ---
    shot_seqs = [torch.from_numpy(item['shot_seq_current']) for item in batch]
    shot_lengths = torch.tensor([len(seq) for seq in shot_seqs], dtype=torch.long)
    padded_shot_seqs = torch.nn.utils.rnn.pad_sequence(shot_seqs, batch_first=True, padding_value=0)

    num_features = padded_shot_seqs.shape[-1]
    B = len(batch)

    # =========================================================================
    # Step A: 預先掃描最大維度
    # =========================================================================

    # L2: Rally History
    l2_max_R = 0
    l2_max_S = 0
    for item in batch:
        rallies = item['rally_history']
        l2_max_R = max(l2_max_R, len(rallies))
        for r in rallies:
            l2_max_S = max(l2_max_S, r['shots'].shape[0])

    # L3: Game History
    l3_max_G = 0
    l3_max_R = 0
    l3_max_S = 0
    for item in batch:
        games = item['game_history']
        l3_max_G = max(l3_max_G, len(games))
        for g in games:
            rallies = g['rallies']
            l3_max_R = max(l3_max_R, len(rallies))
            for r in rallies:
                l3_max_S = max(l3_max_S, r['shots'].shape[0])

    # L4: Set History
    l4_max_St = 0
    l4_max_G = 0
    l4_max_R = 0
    l4_max_S = 0
    for item in batch:
        sets = item['set_history']
        l4_max_St = max(l4_max_St, len(sets))
        for s in sets:
            games = s['games']
            l4_max_G = max(l4_max_G, len(games))
            for g in games:
                rallies = g['rallies']
                l4_max_R = max(l4_max_R, len(rallies))
                for r in rallies:
                    l4_max_S = max(l4_max_S, r['shots'].shape[0])

    # =========================================================================
    # Step B: 預先分配記憶體
    # =========================================================================

    # L2 Tensor: (B, R, S, F)
    l2_max_R = max(l2_max_R, 1)
    l2_max_S = max(l2_max_S, 1)
    l2_tensor = torch.zeros((B, l2_max_R, l2_max_S, num_features), dtype=torch.long)
    l2_lens = torch.zeros((B), dtype=torch.long)
    l2_shot_lens = torch.zeros((B, l2_max_R), dtype=torch.long)

    # L3 Tensor: (B, G, R, S, F)
    l3_max_G = max(l3_max_G, 1)
    l3_max_R = max(l3_max_R, 1)
    l3_max_S = max(l3_max_S, 1)
    l3_tensor = torch.zeros((B, l3_max_G, l3_max_R, l3_max_S, num_features), dtype=torch.long)
    l3_lens = torch.zeros((B), dtype=torch.long)
    l3_rally_lens = torch.zeros((B, l3_max_G), dtype=torch.long)
    l3_shot_lens = torch.zeros((B, l3_max_G, l3_max_R), dtype=torch.long)

    # L4 Tensor: (B, St, G, R, S, F)
    l4_max_St = max(l4_max_St, 1)
    l4_max_G = max(l4_max_G, 1)
    l4_max_R = max(l4_max_R, 1)
    l4_max_S = max(l4_max_S, 1)
    l4_tensor = torch.zeros((B, l4_max_St, l4_max_G, l4_max_R, l4_max_S, num_features), dtype=torch.long)
    l4_lens = torch.zeros((B), dtype=torch.long)
    l4_game_lens = torch.zeros((B, l4_max_St), dtype=torch.long)
    l4_rally_lens = torch.zeros((B, l4_max_St, l4_max_G), dtype=torch.long)
    l4_shot_lens = torch.zeros((B, l4_max_St, l4_max_G, l4_max_R), dtype=torch.long)

    # =========================================================================
    # Step C: 填入數據
    # =========================================================================

    for i, item in enumerate(batch):
        # Fill L2
        rallies = item['rally_history']
        l2_lens[i] = len(rallies)
        for r_idx, r in enumerate(rallies):
            shots = r['shots']
            s_len = shots.shape[0]
            l2_tensor[i, r_idx, :s_len, :] = torch.from_numpy(shots)
            l2_shot_lens[i, r_idx] = s_len

        # Fill L3
        games = item['game_history']
        l3_lens[i] = len(games)
        for g_idx, g in enumerate(games):
            rallies = g['rallies']
            l3_rally_lens[i, g_idx] = len(rallies)
            for r_idx, r in enumerate(rallies):
                shots = r['shots']
                s_len = shots.shape[0]
                l3_tensor[i, g_idx, r_idx, :s_len, :] = torch.from_numpy(shots)
                l3_shot_lens[i, g_idx, r_idx] = s_len

        # Fill L4
        sets = item['set_history']
        l4_lens[i] = len(sets)
        for s_idx, s in enumerate(sets):
            games = s['games']
            l4_game_lens[i, s_idx] = len(games)
            for g_idx, g in enumerate(games):
                rallies = g['rallies']
                l4_rally_lens[i, s_idx, g_idx] = len(rallies)
                for r_idx, r in enumerate(rallies):
                    shots = r['shots']
                    s_len = shots.shape[0]
                    l4_tensor[i, s_idx, g_idx, r_idx, :s_len, :] = torch.from_numpy(shots)
                    l4_shot_lens[i, s_idx, g_idx, r_idx] = s_len

    # --- Targets & Meta ---
    player_ids = torch.tensor([item['player_id'] for item in batch], dtype=torch.long)
    opponent_ids = torch.tensor([item['opponent_id'] for item in batch], dtype=torch.long)
    targets = {key: torch.tensor([item['targets'][key] for item in batch], dtype=torch.long) for key in batch[0]['targets']}

    return {
        # L1
        'shot_seq_current': padded_shot_seqs,
        'shot_seq_current_lengths': shot_lengths,
        # L2
        'rally_history': l2_tensor,
        'rally_history_lengths': l2_lens,
        'rally_history_shot_lengths': l2_shot_lens,
        # L3 (Game History)
        'game_history': l3_tensor,
        'game_history_lengths': l3_lens,
        'game_history_rally_lengths': l3_rally_lens,
        'game_history_shot_lengths': l3_shot_lens,
        # L4 (Set History)
        'set_history': l4_tensor,
        'set_history_lengths': l4_lens,
        'set_history_game_lengths': l4_game_lens,
        'set_history_rally_lengths': l4_rally_lens,
        'set_history_shot_lengths': l4_shot_lens,
        # Meta
        'player_id': player_ids,
        'opponent_id': opponent_ids,
        'targets': targets
    }


# =====================================================================================
# 5. Smart Batch Sampler (適用於 4 層架構，可選)
# =====================================================================================
class SmartBatchSampler(Sampler):
    """
    Smart Batching 採樣器：將長度相近的樣本分在同一個 Batch 中。
    適用於 4 層架構 (Set→Game→Rally→Shot)，也可用於 3 層。
    """
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

        # 根據 sample 中可用的 key 建構排序鍵值
        self.sort_keys = []
        for s in data_source.samples:
            key = (s.get('set_idx', 0), s.get('game_idx', 0), s.get('rally_idx', 0), s.get('target_shot_idx', 0))
            self.sort_keys.append(key)

        indices = np.arange(len(data_source))
        self.sorted_indices = sorted(indices, key=lambda i: self.sort_keys[i])

    def __iter__(self):
        batches = [
            self.sorted_indices[i:i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        np.random.shuffle(batches)
        for batch in batches:
            yield from batch

    def __len__(self):
        return len(self.data_source)


# =====================================================================================
# 6. 工廠函式 — 根據運動自動選擇 Dataset + collate_fn
# =====================================================================================
def get_dataloader(sport_name, data_path, config_path):
    """
    根據運動名稱回傳對應的 (Dataset, collate_fn) 對。

    Args:
        sport_name: 運動名稱 (table_tennis, badminton, tennis)
        data_path: 資料檔案路徑 (e.g. 'table_tennis/processed_data/train_data.pkl')
        config_path: 預處理配置路徑 (e.g. 'table_tennis/processed_data/config.json')

    Returns:
        tuple: (dataset_instance, collate_fn_function)
    """
    from src.default_config import load_sport_config

    # 從運動 YAML 配置取得 hierarchy_levels 和 targets
    sport_config = load_sport_config(sport_name)
    hierarchy_levels = sport_config.get('hierarchy_levels', ['L1', 'L2', 'L3'])
    target_names = sport_config.get('targets', None)
    use_4L = 'L4' in hierarchy_levels

    # 從 YAML 讀取硬上限截斷參數，傳入 Dataset
    max_lens = {
        'max_shot_seq_len':  sport_config.get('max_shot_seq_len',  None),
        'max_rally_seq_len': sport_config.get('max_rally_seq_len', None),
        'max_game_seq_len':  sport_config.get('max_game_seq_len',  None),
        'max_set_seq_len':   sport_config.get('max_set_seq_len',   None),
    }

    if use_4L:
        dataset = PACTDataset4L(data_path, config_path, target_names=target_names, max_lens=max_lens)
        collate = collate_fn_4L
    else:
        dataset = PACTDataset(data_path, config_path, target_names=target_names, max_lens=max_lens)
        collate = collate_fn_3L

    return dataset, collate
