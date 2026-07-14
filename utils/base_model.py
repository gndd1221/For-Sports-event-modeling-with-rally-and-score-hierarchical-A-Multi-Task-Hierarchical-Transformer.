import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """
    所有 Baseline 模型的統一抽象基底類別 (Abstract Base Class)。
    
    這個類別扮演 Adapter (轉接器) 的角色，確保所有的外部或基礎模型
    都能滿足 `train_location_loss.py` 對於輸入輸出合約的要求：
    - 輸入: DataLoader 產生的字典 `batch`
    - 輸出: 各個任務的 Logits 字典 `{task_name: tensor}`
    """
    
    def __init__(self, config):
        """
        初始化 BaseModel 類別。子類別應定義自己的特徵萃取網路，
        然後必須呼叫 `build_task_heads(feature_dim)` 初始化預設的 Task Heads。
        """
        super().__init__()
        self.config = config
        self.task_names = config.get('targets', ['type', 'location', 'strength', 'spin', 'backhand'])
        self.task_heads = nn.ModuleDict()

    @staticmethod
    def _downgrade_4L_to_3L(batch):
        """
        將 4 層 (tennis) batch 降級為 3 層格式，供 Baseline 模型使用。

        4 層 collate_fn_4L 的 set_history 是 6D (B, St, G, R, T, F)，
        但 3 層 baseline 模型預期 5D (B, S, R, T, F)。
        策略：將 Set×Game 展平為 "pseudo-rally" 維度。

        同樣地，set_history_rally_lengths 從 3D (B, St, G) 展平為 2D (B, St×G)，
        set_history_shot_lengths 從 4D (B, St, G, R) 展平為 3D (B, St×G, R)。
        
        若 batch 已經是 3 層格式，直接回傳。
        """
        sh = batch.get('set_history')
        if sh is None or sh.dim() != 6:
            return  # 已經是 3 層格式或沒有 set_history

        B, St, G, R, T, F = sh.shape

        # 6D → 5D: 合併 Set 和 Game 維度成 "pseudo-sets"
        # 將每一個 (Set, Game) 對視為一個獨立的 set-like unit
        batch['set_history'] = sh.reshape(B, St * G, R, T, F)

        # set_history_lengths 表示每筆樣本的有效 set 數量。
        # 現在需要反映 St×G 中有多少個 valid "pseudo-set"
        # 具體做法：sum(game_lengths_per_set)
        sl = batch['set_history_lengths']           # (B,)
        gl = batch['set_history_game_lengths']      # (B, St)

        new_set_lengths = torch.zeros(B, dtype=torch.long, device=sl.device)
        for b in range(B):
            n_sets = sl[b].item()
            total = 0
            for s in range(n_sets):
                total += gl[b, s].item()
            new_set_lengths[b] = total
        batch['set_history_lengths'] = new_set_lengths

        # set_history_rally_lengths: (B, St, G) → (B, St*G)
        srl = batch['set_history_rally_lengths']    # (B, St, G)
        batch['set_history_rally_lengths'] = srl.reshape(B, St * G)

        # set_history_shot_lengths: (B, St, G, R) → (B, St*G, R)
        ssl = batch['set_history_shot_lengths']     # (B, St, G, R)
        batch['set_history_shot_lengths'] = ssl.reshape(B, St * G, R)

    def build_task_heads(self, feature_dim):
        """
        為所有的 Target Tasks 動態建構對應的 Linear Head (分類器)。
        子類別必須在完成自己的 CNN/RNN 建構後呼叫此方法。
        
        Args:
            feature_dim (int): 準備送入線性層的特徵向量維度。
        """
        model_args = self.config.get('model_args', self.config)
        
        for task in self.task_names:
            num_classes = model_args.get(f'num_{task}', self.config.get(f'num_{task}', 2))
            self.task_heads[f'head_{task}'] = nn.Linear(feature_dim, num_classes)

    @abstractmethod
    def extract_features(self, batch):
        """
        [必須實作]
        處理輸入 Tensor `batch` 並生成長度為 `feature_dim` 的向量。
        """
        pass

    def forward(self, batch):
        """
        [標準合約介面]
        整合 Forward Pass。會先呼叫 `extract_features`，然後將特徵轉換成 logits 字典。
        """
        features = self.extract_features(batch)
        
        logits_dict = {}
        for task in self.task_names:
            head_key = f'head_{task}'
            if head_key in self.task_heads:
                logits_dict[task] = self.task_heads[head_key](features)
                
        return logits_dict
