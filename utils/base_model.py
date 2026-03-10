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
