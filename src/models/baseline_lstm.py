import torch
import torch.nn as nn
import sys
import os

# 確保專案根目錄在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.base_model import BaseModel
from src.model_components import PlayerStyleEncoder

class BaselineLSTM(BaseModel):
    """
    經典的 Baseline 模型：雙向 LSTM (Bi-directional LSTM)。
    
    這個模型會接收我們定義的 DataLoader Input Dict `batch`，
    透過 Player Embedding 以及類別特徵的 Embedding 計算出特徵後，
    送入 LSTM 層，取出最後一個時間點的有效 Hidden State，
    最後呼叫繼承自 `BaseModel` 的 `task_heads` 完成跨任務預測。
    """
    
    def __init__(self, config):
        """
        初始化 LSTM 模型並建構對應的任務頭 (Task Heads)。
        """
        # 注意：BaseModel 的 __init__ 已準備好了 task_names 空殼。
        super().__init__(config)
        
        self.config = config
        model_args = config.get('model_args', config)
        self.d_model = model_args.get('d_model', 128)
        self.hidden_dim = model_args.get('hidden_dim', 256)
        self.num_layers = model_args.get('num_layers', 2)
        self.dropout = model_args.get('dropout', 0.3)
        self.player_embedding_dim = model_args.get('player_embedding_dim', 32)
        
        # 尋找映射表 Index
        self.feature_indices = {name: i for i, name in enumerate(config['features_to_extract'])}
        
        # 建立類別特徵 Embedding Layers
        self.player_encoder = PlayerStyleEncoder(config['num_players'], self.player_embedding_dim)
        
        self.embedding_layers = nn.ModuleDict()
        other_embedding_dim = 0
        for feat_name in config['categorical_features']:
            vocab_size = config.get('vocab_sizes', {}).get(feat_name, config.get(f'num_{feat_name}', 2))
            embedding_dim = config.get('embedding_dims', {}).get(feat_name, 8)
            safe_key = f"{feat_name}_embedding"
            self.embedding_layers[safe_key] = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            other_embedding_dim += embedding_dim
            
        # 輸入特徵總維度
        # player_id + opponent_id + categorical embeddings + numerical features
        self.input_dim = (self.player_embedding_dim * 2) + other_embedding_dim + len(config['numerical_features'])
        
        # LSTM 本體
        # 使用 batch_first=True, 因為 input_shape = (Batch, SeqLen, FeatureDim)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        # Bi-LSTM 輸出為 hidden_dim * 2
        lstm_out_dim = self.hidden_dim * 2
        
        # (重要) 必須呼叫父類別方法建立 Task Heads，否則會報錯
        self.build_task_heads(feature_dim=lstm_out_dim)

    def _embed_and_combine_features(self, raw_features):
        """將原始序列資料轉換為嵌入特徵 (與 PACT 雷同的預處理)"""
        all_parts = []

        # 1. 處理 Player/Opponent
        player_id_col = self.feature_indices['player_id']
        opponent_id_col = self.feature_indices['opponent_id']
        player_ids = raw_features[..., player_id_col].long()
        opponent_ids = raw_features[..., opponent_id_col].long()
        
        player_emb = self.player_encoder.embedding(player_ids)
        opponent_emb = self.player_encoder.embedding(opponent_ids)
        all_parts.extend([player_emb, opponent_emb])

        # 2. 處理 Categorical 
        for feat_name in self.config['categorical_features']:
            col_idx = self.feature_indices[feat_name]
            feat_values = raw_features[..., col_idx].long()
            safe_key = f"{feat_name}_embedding"
            embedded = self.embedding_layers[safe_key](feat_values)
            all_parts.append(embedded)

        # 3. 處理 Numerical
        num_feat_indices = [self.feature_indices[name] for name in self.config['numerical_features']]
        # 確保 Numerical 取出後仍有 Feature Dim (..., 1) 的形狀
        if len(num_feat_indices) > 0:
            numerical_features = raw_features[..., num_feat_indices].float()
            all_parts.append(numerical_features)

        # 統整最後維度 (Batch, SeqLen, Input_Dim)
        combined = torch.cat(all_parts, dim=-1)
        return combined

    def extract_features(self, batch):
        """
        [實作 BaseModel 合約]
        將資料送入 LSTM，並返回序列中對應最後一個有效長度的特徵向量。
        """
        # --- 取出 L1 的最新球種序列集 (`shot_seq_current`) 作為預測基礎 ---
        # 如果有需要融合 Context (L2/L3)，子類需修改資料長度準備方式，此處做最簡 Baseline
        # 將輸入搬移到跟模型相同的 Device
        device = next(self.parameters()).device
        shot_seq_current_raw = batch['shot_seq_current'].to(device)            # (Batch, SeqLen, Features)
        shot_seq_lengths = batch['shot_seq_current_lengths'].cpu()  # 打包 LSTM 時需要 CPU

        # 特徵轉換
        x_embedded = self._embed_and_combine_features(shot_seq_current_raw)
        
        # 通過 LSTM。為了支援 Padding，我們可以用 pack_padded_sequence 加速
        # 但直接塞也行，只是要記得用 lengths 抓出對應最後一步。
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x_embedded, 
            lengths=shot_seq_lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # out 形狀 (Batch, SeqLen, Hidden * 2)
        
        # 取每一個 Batch 的最後一個有效長度 Output (因為打擊序列長短不一)
        batch_size = out.size(0)
        # valid_lengths 最大值不能小於 1 (防呆)
        valid_lengths = torch.clamp(shot_seq_lengths - 1, min=0).to(out.device)
        
        # 透過 gather 取出真正的尾端時間步特徵
        index = valid_lengths.view(batch_size, 1, 1).expand(batch_size, 1, out.size(-1))
        last_hidden_state = out.gather(1, index).squeeze(1)
        
        return last_hidden_state
