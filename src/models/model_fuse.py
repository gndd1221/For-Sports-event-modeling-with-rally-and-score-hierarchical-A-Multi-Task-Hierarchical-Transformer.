"""
model_fuse.py — 統一的 PACT-iTransformer 模型

透過 config 中的兩個參數控制所有模型變體：
  - hierarchy_levels: ['L1'], ['L1','L2'], ['L1','L2','L3'], ['L1','L2','L3','L4']
  - fusion_type: 'cls_token', 'task_project', 'task_attention'
  - use_sequence_fusion: True/False (sequence_attention 模式)

對應關係：
  parallel             = hierarchy=['L1','L2','L3'] + fusion='cls_token'
  task_project         = hierarchy=['L1','L2','L3'] + fusion='task_project'
  task_attention       = hierarchy=['L1','L2','L3'] + fusion='task_attention'
  sequence_attention   = hierarchy=['L1','L2','L3'] + fusion='task_attention' + use_sequence_fusion=True
  L1_L2                = hierarchy=['L1','L2']       + fusion='cls_token'
  L1                   = hierarchy=['L1']            + fusion='cls_token'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 確保專案根目錄在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.model_components import (
    HierarchicalEncoder,
    PlayerStyleEncoder,
    CLSTokenFusion,
    TaskProjectionFusion,
    TaskAttentionFusion,
    DynamicPredictionHead,
    AttentionPooling,
    TopDownRefinement,
)


class PACTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_args = config.get('model_args', config)
        d_model = model_args['d_model']
        nhead = model_args['nhead']
        num_encoder_layers = model_args['num_encoder_layers']
        dim_feedforward = model_args['dim_feedforward']
        dropout = model_args['dropout']
        player_embedding_dim = model_args['player_embedding_dim']

        # --- 從 config 讀取模型變體控制參數 ---
        self.hierarchy_levels = model_args.get('hierarchy_levels', ['L1', 'L2', 'L3'])
        self.fusion_type = model_args.get('fusion_type', 'cls_token')
        self.use_L2 = 'L2' in self.hierarchy_levels
        self.use_L3 = 'L3' in self.hierarchy_levels
        self.use_L4 = 'L4' in self.hierarchy_levels
        self.use_sequence_fusion = model_args.get('use_sequence_fusion', False)

        self.feature_indices = {name: i for i, name in enumerate(config['features_to_extract'])}

        # --- 序列聚合策略與預測頭深度 ---
        self.pooling_type = model_args.get('pooling_type', 'last')  # 'last', 'mean', 'attention'
        self.head_depth = model_args.get('head_depth', 1)

        # --- 球員編碼器 ---
        self.player_encoder = PlayerStyleEncoder(config['num_players'], player_embedding_dim)

        # --- (A) 特徵嵌入層 ---
        self.embedding_layers = nn.ModuleDict()
        other_embedding_dim = 0
        for feat_name in config['categorical_features']:
            vocab_size = config['vocab_sizes'][feat_name]
            embedding_dim = config['embedding_dims'][feat_name]
            safe_key = f"{feat_name}_embedding"
            self.embedding_layers[safe_key] = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            other_embedding_dim += embedding_dim

        projector_input_dim = (player_embedding_dim * 2) + other_embedding_dim + len(config['numerical_features'])

        # --- (B) 輸入投影 ---
        self.input_projection = nn.Sequential(
            nn.Linear(projector_input_dim, d_model),
            nn.LayerNorm(d_model)
        )

        # --- (B2) Skip Connection — 多拍聚合原始特徵 ---
        self.skip_window_size = model_args.get('skip_window_size', 0)
        self.use_skip_connection = self.skip_window_size > 0
        if self.use_skip_connection:
            self.skip_proj = nn.Linear(projector_input_dim, d_model)
            self.skip_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )

        # --- (B3) Shot-Aware Positional Encoding (SOPE) ---
        self.use_shot_aware_pe = model_args.get('use_shot_aware_pe', False)
        if self.use_shot_aware_pe:
            self.is_serve_embedding = nn.Embedding(2, d_model)   # 0=非發球, 1=發球
            self.is_self_embedding = nn.Embedding(2, d_model)    # 0=對手拍, 1=我方拍

        # --- (C) PACT 編碼器 (按需建立) ---
        self.shot_encoder = HierarchicalEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        if self.use_L2:
            self.rally_encoder = HierarchicalEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        if self.use_L3 and not self.use_L4:
            # 3 層 (L1+L2+L3): L3 = Set (直接從 Rally 聚合)
            self.highest_encoder = HierarchicalEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        if self.use_L4:
            # 4 層 (L1+L2+L3+L4): L3 = Game, L4 = Set
            self.game_encoder = HierarchicalEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
            self.set_encoder = HierarchicalEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)

        # --- (參) Attention Pooling (按需建立) ---
        if self.pooling_type == 'attention':
            self.attn_pool = AttentionPooling(d_model, nhead=1, dropout=dropout)

        # --- (D) iTransformer 路徑 (按需建立) ---
        # D.1 L1 (Shot) — 總是存在
        self.max_shot_seq_len = config['max_shot_seq_len']
        self.itrans_shot_embedding = nn.Linear(self.max_shot_seq_len, d_model)
        self.itrans_shot_norm = nn.LayerNorm(d_model)
        itrans_shot_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
        self.itrans_shot_encoder = nn.TransformerEncoder(itrans_shot_layer, num_encoder_layers)

        # D.2 L2 (Rally)
        if self.use_L2:
            self.max_rally_seq_len = config['max_rally_seq_len']
            self.itrans_rally_embedding = nn.Linear(self.max_rally_seq_len, d_model)
            self.itrans_rally_norm = nn.LayerNorm(d_model)
            itrans_rally_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
            self.itrans_rally_encoder = nn.TransformerEncoder(itrans_rally_layer, num_encoder_layers)

        # D.3 L3: 3層時用 max_set_seq_len (highest), 4層時用 max_game_seq_len
        if self.use_L3 and not self.use_L4:
            self.max_highest_seq_len = config['max_set_seq_len']
            self.itrans_highest_embedding = nn.Linear(self.max_highest_seq_len, d_model)
            self.itrans_highest_norm = nn.LayerNorm(d_model)
            itrans_highest_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
            self.itrans_highest_encoder = nn.TransformerEncoder(itrans_highest_layer, num_encoder_layers)
        if self.use_L4:
            self.max_game_seq_len = config['max_game_seq_len']
            self.itrans_game_embedding = nn.Linear(self.max_game_seq_len, d_model)
            self.itrans_game_norm = nn.LayerNorm(d_model)
            itrans_game_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
            self.itrans_game_encoder = nn.TransformerEncoder(itrans_game_layer, num_encoder_layers)

            self.max_set_seq_len = config['max_set_seq_len']
            self.itrans_set_embedding = nn.Linear(self.max_set_seq_len, d_model)
            self.itrans_set_norm = nn.LayerNorm(d_model)
            itrans_set_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True)
            self.itrans_set_encoder = nn.TransformerEncoder(itrans_set_layer, num_encoder_layers)

        # --- (E) PACT + iTransformer 融合投影 (按需建立) ---
        self.use_gated_fusion = model_args.get('use_gated_fusion', False)

        # 根據是否使用門控融合，建立不同的融合元件
        if self.use_gated_fusion:
            # 門控融合：Sigmoid Gate 動態混合 PACT 與 iTransformer
            self.shot_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
            self.shot_combination_norm = nn.LayerNorm(d_model)
            if self.use_L2:
                self.rally_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
                self.rally_combination_norm = nn.LayerNorm(d_model)
            if self.use_L3 and not self.use_L4:
                self.highest_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
                self.highest_combination_norm = nn.LayerNorm(d_model)
            if self.use_L4:
                self.game_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
                self.game_combination_norm = nn.LayerNorm(d_model)
                self.set_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
                self.set_combination_norm = nn.LayerNorm(d_model)
        else:
            # 原始融合：Concatenate + Linear 投影
            self.shot_combination_proj = nn.Linear(d_model * 2, d_model)
            self.shot_combination_norm = nn.LayerNorm(d_model)
            if self.use_L2:
                self.rally_combination_proj = nn.Linear(d_model * 2, d_model)
                self.rally_combination_norm = nn.LayerNorm(d_model)
            if self.use_L3 and not self.use_L4:
                self.highest_combination_proj = nn.Linear(d_model * 2, d_model)
                self.highest_combination_norm = nn.LayerNorm(d_model)
            if self.use_L4:
                self.game_combination_proj = nn.Linear(d_model * 2, d_model)
                self.game_combination_norm = nn.LayerNorm(d_model)
                self.set_combination_proj = nn.Linear(d_model * 2, d_model)
                self.set_combination_norm = nn.LayerNorm(d_model)

        # --- (F) 由上而下的注意力引導 (Top-Down Refinement) ---
        self.use_top_down_attention = model_args.get('use_top_down_attention', False)
        if self.use_top_down_attention:
            self.td_refinement = TopDownRefinement(d_model, nhead, dim_feedforward, dropout)
            # 使用可學習的 0 初始化 Gate，確保訓練初期等同於關閉，只專注於穩定 L1 摘要
            self.td_gate = nn.Parameter(torch.zeros(1))

        # --- (H) Turn-Based Style Gating (TBSG) ---
        self.use_turn_based_gating = model_args.get('use_turn_based_gating', False)
        # 注意: 我們改為採用無參數的 Hard Swap 機制，因此不需要在此建立 Neural Network Layer

        # --- (I) Temporal-Scale Adaptive Gating (TSAG) ---
        self.use_temporal_scale_gating = model_args.get('use_temporal_scale_gating', False)
        if self.use_temporal_scale_gating:
            # 計算實際階層數量
            num_levels = 1  # L1 永遠存在
            if self.use_L2:
                num_levels += 1
            if self.use_L3 and not self.use_L4:
                num_levels += 1
            if self.use_L4:
                num_levels += 2  # game + set
            self.tsag_num_levels = num_levels
            
            # 取得任務列表
            self.tsag_task_names = config.get('targets', ['type', 'backhand', 'location', 'strength', 'spin'])
            self.tsag_num_tasks = len(self.tsag_task_names)
            
            # 拍數嵌入: 將拍數編碼為連續向量
            self.tsag_length_embedding = nn.Embedding(256, d_model)
            # Gate 網路: 輸出各任務專屬的階層權重 (softmax 前的 logits)
            # 修改: 引入 Content-Aware (接收 length_emb + h_shot_last)
            self.tsag_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, self.tsag_num_tasks * num_levels)
            )
            # 關鍵安全門: 將最後一層初始化為 0
            # Sigmoid(0)*2 = 1.0 → 訓練初期等同於不加權
            nn.init.zeros_(self.tsag_gate[-1].weight)
            nn.init.zeros_(self.tsag_gate[-1].bias)

        # --- (G) 最終融合模組 (依 fusion_type 建立) ---
        task_names = config.get('targets', ['type', 'backhand', 'location', 'strength', 'spin'])

        if self.fusion_type == 'cls_token':
            self.fusion_module = CLSTokenFusion(
                d_model=d_model,
                player_embedding_dim=player_embedding_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_fusion_layers=1
            )
        elif self.fusion_type == 'task_project':
            self.fusion_module = TaskProjectionFusion(
                d_model=d_model,
                player_embedding_dim=player_embedding_dim,
                task_names=task_names,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_fusion_layers=1
            )
        elif self.fusion_type == 'task_attention':
            num_fusion_layers = model_args.get('num_fusion_layers', 1)
            self.fusion_module = TaskAttentionFusion(
                d_model=d_model,
                player_embedding_dim=player_embedding_dim,
                task_names=task_names,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                num_fusion_layers=num_fusion_layers,
                use_task_decoder=model_args.get('use_task_decoder', False)
            )
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}. "
                             f"Must be 'cls_token', 'task_project', or 'task_attention'.")

        # --- (G) 預測頭 (動態建構，支援任意運動) ---
        task_configs = {
            task: config[f'num_{task}']
            for task in task_names
        }
        self.prediction_head = DynamicPredictionHead(
            d_model, task_configs,
            head_depth=self.head_depth,
            dropout=dropout
        )

        # --- (H) 空白摘要 (用於 padding，按需建立) ---
        if self.use_L2:
            self.empty_rally_summary = nn.Parameter(torch.zeros(d_model))
        if self.use_L3 and not self.use_L4:
            self.empty_highest_summary = nn.Parameter(torch.zeros(d_model))
        if self.use_L4:
            self.empty_game_summary = nn.Parameter(torch.zeros(d_model))
            self.empty_set_summary = nn.Parameter(torch.zeros(d_model))

    # ========================================================================
    # 輔助方法
    # ========================================================================

    def _embed_and_combine_features(self, raw_features):
        """嵌入並組合所有原始特徵"""
        all_parts = []

        player_id_col = self.feature_indices['player_id']
        opponent_id_col = self.feature_indices['opponent_id']
        player_ids = raw_features[..., player_id_col].long()
        opponent_ids = raw_features[..., opponent_id_col].long()
        player_emb = self.player_encoder.embedding(player_ids)
        opponent_emb = self.player_encoder.embedding(opponent_ids)
        all_parts.extend([player_emb, opponent_emb])

        for feat_name in self.config['categorical_features']:
            col_idx = self.feature_indices[feat_name]
            feat_values = raw_features[..., col_idx].long()
            safe_key = f"{feat_name}_embedding"
            embedded = self.embedding_layers[safe_key](feat_values)
            all_parts.append(embedded)

        num_feat_indices = [self.feature_indices[name] for name in self.config['numerical_features']]
        numerical_features = raw_features[..., num_feat_indices].float()
        all_parts.append(numerical_features)

        combined = torch.cat(all_parts, dim=-1)
        return combined

    def _run_itrans_hierarchical_path(self, x_in, max_len, embedding_layer, norm_layer, encoder):
        """運行 iTransformer 的階層式路徑"""
        B, L_batch_max, D_model = x_in.shape
        L_config_max = max_len

        x_padded = torch.zeros(B, L_config_max, D_model,
                               device=x_in.device, dtype=x_in.dtype)
        copy_len = min(L_batch_max, L_config_max)
        x_padded[:, :copy_len, :] = x_in[:, :copy_len, :]

        x_permuted = x_padded.permute(0, 2, 1)
        x_emb = embedding_layer(x_permuted)
        x_emb_norm = norm_layer(x_emb)
        h_seq = encoder(x_emb_norm)
        h_last = torch.mean(h_seq, dim=1)

        return h_last

    def _create_padding_mask(self, lengths, max_len):
        """創建 padding mask"""
        batch_size = lengths.size(0)
        mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
        return mask

    def _get_last_valid_output(self, sequence_output, lengths):
        """獲取序列的最後一個有效輸出"""
        batch_size = sequence_output.size(0)
        valid_lengths = torch.clamp(lengths - 1, min=0)
        index = valid_lengths.view(batch_size, 1, 1).expand(-1, -1, sequence_output.size(-1))
        return sequence_output.gather(1, index).squeeze(1)

    def _get_pooled_output(self, sequence_output, padding_mask):
        """執行遮罩平均池化"""
        mask = padding_mask.unsqueeze(-1).expand_as(sequence_output)
        pooled_output = sequence_output.clone()
        pooled_output[mask] = 0.0
        sum_pooled = torch.sum(pooled_output, dim=1)
        valid_counts = (~padding_mask).sum(dim=1).float().unsqueeze(-1)
        valid_counts = torch.clamp(valid_counts, min=1.0)
        mean_pooled = sum_pooled / valid_counts
        return mean_pooled

    def _get_summary(self, sequence_output, lengths, padding_mask=None):
        """根據 pooling_type 選擇聚合策略"""
        if self.pooling_type == 'attention':
            if padding_mask is None:
                max_len = sequence_output.size(1)
                padding_mask = self._create_padding_mask(lengths, max_len)
            return self.attn_pool(sequence_output, padding_mask=padding_mask)
        elif self.pooling_type == 'mean':
            if padding_mask is None:
                max_len = sequence_output.size(1)
                padding_mask = self._create_padding_mask(lengths, max_len)
            return self._get_pooled_output(sequence_output, padding_mask)
        else:  # 'last' (預設)
            return self._get_last_valid_output(sequence_output, lengths)

    def _combine_pact_itrans(self, h_pact, h_itrans, gate_or_proj, norm):
        """
        融合 PACT 與 iTransformer 兩條路徑的特徵。
        - 門控模式 (use_gated_fusion=True): gate = sigmoid(concat) → g * pact + (1-g) * itrans
        - 投影模式 (原始):                  concat → linear_proj → d_model
        """
        combined = torch.cat([h_pact, h_itrans], dim=-1)  # (B, d_model*2)
        if self.use_gated_fusion:
            gate = gate_or_proj(combined)  # (B, d_model) — 0~1 之間
            fused = gate * h_pact + (1 - gate) * h_itrans
        else:
            fused = gate_or_proj(combined)  # Linear: (B, d_model*2) → (B, d_model)
        return norm(fused)

    def _apply_skip_connection(self, fusion_output, last_shot_proj):
        """
        將最後一拍的原始嵌入透過 Gated Skip Connection 注入 Fusion 輸出。
        支援 tensor (cls_token) 和 dict (task_project/task_attention) 兩種格式。
        """
        if isinstance(fusion_output, dict):
            # task_project / task_attention: {task_name: (B, d_model)}
            gated_output = {}
            for task_name, feat in fusion_output.items():
                gate = self.skip_gate(torch.cat([feat, last_shot_proj], dim=-1))
                gated_output[task_name] = gate * feat + (1 - gate) * last_shot_proj
            return gated_output
        else:
            # cls_token: (B, d_model)
            gate = self.skip_gate(torch.cat([fusion_output, last_shot_proj], dim=-1))
            return gate * fusion_output + (1 - gate) * last_shot_proj

    # ========================================================================
    # Forward
    # ========================================================================

    def forward(self, batch):
        device = next(self.parameters()).device

        # --- 移動資料到 device ---
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, dict):
                for k, v in value.items():
                    batch[key][k] = v.to(device)

        player_id = batch['player_id']
        opponent_id = batch['opponent_id']
        B_main = player_id.shape[0]

        # 收集各階層的最終特徵向量
        hierarchy_features = []

        # ===== L4 (Set History) — 4 層: Shot→Rally→Game→Set =====
        if self.use_L4:
            set_history_lengths = batch['set_history_lengths']
            if set_history_lengths.max() > 0:
                set_history_raw = batch['set_history']
                B, S, G, R, T, F = set_history_raw.shape  # 6D
                set_history_embedded = self._embed_and_combine_features(set_history_raw.view(-1, F))
                set_history_proj = self.input_projection(set_history_embedded).view(B, S, G, R, T, -1)

                shots_flat = set_history_proj.view(B * S * G * R, T, -1)
                shot_lengths_flat = batch['set_history_shot_lengths'].view(-1)
                shot_mask = self._create_padding_mask(shot_lengths_flat, T)
                encoded_shots = self.shot_encoder(shots_flat, shot_mask, lengths=shot_lengths_flat)
                rally_summaries = self._get_last_valid_output(encoded_shots, shot_lengths_flat)

                rallies_flat = rally_summaries.view(B * S * G, R, -1)
                rally_lengths_flat = batch['set_history_rally_lengths'].view(-1)
                rally_mask = self._create_padding_mask(rally_lengths_flat, R)
                encoded_rallies = self.rally_encoder(rallies_flat, rally_mask, lengths=rally_lengths_flat)
                game_summaries = self._get_last_valid_output(encoded_rallies, rally_lengths_flat)

                games_flat = game_summaries.view(B * S, G, -1)
                game_lengths_flat = batch['set_history_game_lengths'].view(-1)
                game_mask = self._create_padding_mask(game_lengths_flat, G)
                encoded_games = self.game_encoder(games_flat, game_mask, lengths=game_lengths_flat)
                set_summaries = self._get_last_valid_output(encoded_games, game_lengths_flat)

                sets_flat = set_summaries.view(B, S, -1)
                set_mask = self._create_padding_mask(set_history_lengths, S)
                h_set_sequence = self.set_encoder(sets_flat, set_mask, lengths=set_history_lengths)
                h_set_last_pact = self._get_summary(h_set_sequence, set_history_lengths, set_mask)

                h_set_last_itrans = self._run_itrans_hierarchical_path(
                    sets_flat, self.max_set_seq_len,
                    self.itrans_set_embedding, self.itrans_set_norm, self.itrans_set_encoder
                )
                h_set_last = self._combine_pact_itrans(
                    h_set_last_pact, h_set_last_itrans,
                    self.set_gate if self.use_gated_fusion else self.set_combination_proj,
                    self.set_combination_norm
                )
            else:
                h_set_last = self.empty_set_summary.unsqueeze(0).expand(B_main, -1)

        # ===== L3 — 雙模式 =====
        if self.use_L3 and self.use_L4:
            # 4 層模式: L3 = Game History (5D: B,G,R,T,F)
            game_history_lengths = batch['game_history_lengths']
            if game_history_lengths.max() > 0:
                game_history_raw = batch['game_history']
                B, G, R, T, F = game_history_raw.shape
                game_history_embedded = self._embed_and_combine_features(game_history_raw.view(-1, F))
                game_history_proj = self.input_projection(game_history_embedded).view(B, G, R, T, -1)

                shots_flat = game_history_proj.view(B * G * R, T, -1)
                shot_lengths_flat = batch['game_history_shot_lengths'].view(-1)
                shot_mask = self._create_padding_mask(shot_lengths_flat, T)
                encoded_shots = self.shot_encoder(shots_flat, shot_mask, lengths=shot_lengths_flat)
                rally_summaries = self._get_last_valid_output(encoded_shots, shot_lengths_flat)

                rallies_flat = rally_summaries.view(B * G, R, -1)
                rally_lengths_flat = batch['game_history_rally_lengths'].view(-1)
                rally_mask = self._create_padding_mask(rally_lengths_flat, R)
                encoded_rallies = self.rally_encoder(rallies_flat, rally_mask, lengths=rally_lengths_flat)
                game_summaries = self._get_last_valid_output(encoded_rallies, rally_lengths_flat)

                games_flat = game_summaries.view(B, G, -1)
                game_mask = self._create_padding_mask(game_history_lengths, G)
                h_game_sequence = self.game_encoder(games_flat, game_mask, lengths=game_history_lengths)
                h_game_last_pact = self._get_summary(h_game_sequence, game_history_lengths, game_mask)

                h_game_last_itrans = self._run_itrans_hierarchical_path(
                    games_flat, self.max_game_seq_len,
                    self.itrans_game_embedding, self.itrans_game_norm, self.itrans_game_encoder
                )
                h_game_last = self._combine_pact_itrans(
                    h_game_last_pact, h_game_last_itrans,
                    self.game_gate if self.use_gated_fusion else self.game_combination_proj,
                    self.game_combination_norm
                )
            else:
                h_game_last = self.empty_game_summary.unsqueeze(0).expand(B_main, -1)

        elif self.use_L3 and not self.use_L4:
            # 3 層模式: L3 = Set History (5D: B,S,R,T,F) — 原始行為
            set_history_lengths = batch['set_history_lengths']
            if set_history_lengths.max() > 0:
                set_history_raw = batch['set_history']
                B, S, R, T, F = set_history_raw.shape
                set_history_embedded = self._embed_and_combine_features(set_history_raw.view(-1, F))
                set_history_proj = self.input_projection(set_history_embedded).view(B, S, R, T, -1)
                set_history_shots = set_history_proj.view(B * S * R, T, -1)
                set_history_shot_lengths = batch['set_history_shot_lengths'].view(-1)
                shot_mask = self._create_padding_mask(set_history_shot_lengths, T)

                encoded_shots = self.shot_encoder(set_history_shots, shot_mask, lengths=set_history_shot_lengths)
                rally_summaries = self._get_last_valid_output(encoded_shots, set_history_shot_lengths)
                set_history_rallies = rally_summaries.view(B * S, R, -1)
                set_history_rally_lengths = batch['set_history_rally_lengths'].view(-1)
                rally_mask = self._create_padding_mask(set_history_rally_lengths, R)
                encoded_rallies = self.rally_encoder(set_history_rallies, rally_mask, lengths=set_history_rally_lengths)
                set_summaries = self._get_last_valid_output(encoded_rallies, set_history_rally_lengths)
                set_history_sets = set_summaries.view(B, S, -1)
                set_mask = self._create_padding_mask(set_history_lengths, S)
                h_highest_sequence = self.highest_encoder(set_history_sets, set_mask, lengths=set_history_lengths)
                h_highest_last_pact = self._get_summary(h_highest_sequence, set_history_lengths, set_mask)

                h_highest_last_itrans = self._run_itrans_hierarchical_path(
                    set_history_sets, self.max_highest_seq_len,
                    self.itrans_highest_embedding, self.itrans_highest_norm, self.itrans_highest_encoder
                )
                h_highest_last = self._combine_pact_itrans(
                    h_highest_last_pact, h_highest_last_itrans,
                    self.highest_gate if self.use_gated_fusion else self.highest_combination_proj,
                    self.highest_combination_norm
                )
            else:
                h_highest_last = self.empty_highest_summary.unsqueeze(0).expand(B_main, -1)

        # ===== L2 (Rally History) =====
        if self.use_L2:
            rally_history_lengths = batch['rally_history_lengths']
            if rally_history_lengths.max() > 0:
                rally_history_raw = batch['rally_history']
                B, R, T, F = rally_history_raw.shape
                rally_history_embedded = self._embed_and_combine_features(rally_history_raw.view(-1, F))
                rally_history_proj = self.input_projection(rally_history_embedded).view(B, R, T, -1)
                rally_history_shots = rally_history_proj.view(B * R, T, -1)
                rally_history_shot_lengths = batch['rally_history_shot_lengths'].view(-1)
                shot_mask = self._create_padding_mask(rally_history_shot_lengths, T)

                encoded_shots_rally = self.shot_encoder(rally_history_shots, shot_mask, lengths=rally_history_shot_lengths)
                rally_summaries = self._get_last_valid_output(encoded_shots_rally, rally_history_shot_lengths)
                rally_history_rallies = rally_summaries.view(B, R, -1)
                rally_mask = self._create_padding_mask(rally_history_lengths, R)
                h_rally_sequence = self.rally_encoder(rally_history_rallies, rally_mask, lengths=rally_history_lengths)
                h_rally_last_pact = self._get_summary(h_rally_sequence, rally_history_lengths, rally_mask)

                h_rally_last_itrans = self._run_itrans_hierarchical_path(
                    rally_history_rallies, self.max_rally_seq_len,
                    self.itrans_rally_embedding, self.itrans_rally_norm, self.itrans_rally_encoder
                )
                h_rally_last = self._combine_pact_itrans(
                    h_rally_last_pact, h_rally_last_itrans,
                    self.rally_gate if self.use_gated_fusion else self.rally_combination_proj,
                    self.rally_combination_norm
                )
            else:
                h_rally_last = self.empty_rally_summary.unsqueeze(0).expand(B_main, -1)

        # --- 處理空歷史的 fallback ---
        if self.use_L4:
            set_mask_empty = (batch['set_history_lengths'] == 0).unsqueeze(1).expand_as(h_set_last)
            h_set_last = torch.where(set_mask_empty, self.empty_set_summary, h_set_last)
            game_mask_empty = (batch['game_history_lengths'] == 0).unsqueeze(1).expand_as(h_game_last)
            h_game_last = torch.where(game_mask_empty, self.empty_game_summary, h_game_last)
        elif self.use_L3: # This implies use_L3 and not use_L4
            highest_mask_empty = (batch['set_history_lengths'] == 0).unsqueeze(1).expand_as(h_highest_last)
            h_highest_last = torch.where(highest_mask_empty, self.empty_highest_summary, h_highest_last)
        if self.use_L2:
            rally_mask_empty = (batch['rally_history_lengths'] == 0).unsqueeze(1).expand_as(h_rally_last)
            h_rally_last = torch.where(rally_mask_empty, self.empty_rally_summary, h_rally_last)

        # ===== L1 (Current Shot Sequence) — 總是執行 =====
        shot_seq_current_raw = batch['shot_seq_current']
        shot_seq_current_embedded = self._embed_and_combine_features(shot_seq_current_raw)
        shot_seq_current_lengths = batch['shot_seq_current_lengths']

        # Skip Connection: 擷取最後一拍的原始嵌入 (尚未經 Transformer 編碼)
        if self.use_skip_connection:
            last_shot_idx = torch.clamp(shot_seq_current_lengths - 1, min=0)
            last_shot_raw = shot_seq_current_embedded[
                torch.arange(B_main, device=device), last_shot_idx
            ]  # (B, projector_input_dim)
            last_shot_proj = self.skip_proj(last_shot_raw)  # (B, d_model)

        # Path A: PACT
        shot_seq_current_proj = self.input_projection(shot_seq_current_embedded)

        # Shot-Aware Positional Encoding: 注入「發球/非發球」與「我方/對手」的語義嵌入
        if self.use_shot_aware_pe:
            T_cur = shot_seq_current_proj.shape[1]
            positions = torch.arange(T_cur, device=device).unsqueeze(0).expand(B_main, -1)
            is_serve = (positions == 0).long()  # 第 0 拍 = 發球
            # 判定「我方拍」：比較每一拍的 player_id 與 batch 的主球員 player_id
            player_id_col = self.feature_indices['player_id']
            shot_player_ids = shot_seq_current_raw[..., player_id_col].long()  # (B, T)
            batch_player_id = player_id.unsqueeze(1).expand_as(shot_player_ids)  # (B, T)
            is_self = (shot_player_ids == batch_player_id).long()
            shot_seq_current_proj = shot_seq_current_proj + self.is_serve_embedding(is_serve) + self.is_self_embedding(is_self)

        shot_mask = self._create_padding_mask(shot_seq_current_lengths, shot_seq_current_proj.shape[1])
        h_shot_sequence = self.shot_encoder(shot_seq_current_proj, shot_mask, lengths=shot_seq_current_lengths)
        h_shot_last_pact = self._get_summary(h_shot_sequence, shot_seq_current_lengths, shot_mask)

        # Path B: iTransformer
        x_for_itrans = shot_seq_current_embedded
        B, S_batch_max, F_combined = x_for_itrans.shape
        S_config_max = self.max_shot_seq_len

        x_itrans_padded = torch.zeros(B, S_config_max, F_combined,
                                      device=x_for_itrans.device, dtype=x_for_itrans.dtype)
        copy_len = min(S_batch_max, S_config_max)
        x_itrans_padded[:, :copy_len, :] = x_for_itrans[:, :copy_len, :]

        x_itrans_permuted = x_itrans_padded.permute(0, 2, 1)
        x_itrans_emb = self.itrans_shot_embedding(x_itrans_permuted)
        x_itrans_emb_norm = self.itrans_shot_norm(x_itrans_emb)
        h_itrans_seq = self.itrans_shot_encoder(x_itrans_emb_norm)
        h_shot_last_itrans = torch.mean(h_itrans_seq, dim=1)

        # Fuse PACT + iTransformer for L1
        h_shot_last = self._combine_pact_itrans(
            h_shot_last_pact, h_shot_last_itrans,
            self.shot_gate if self.use_gated_fusion else self.shot_combination_proj,
            self.shot_combination_norm
        )

        # --- Top-Down Refinement ---
        if self.use_top_down_attention:
            if self.use_L4:
                high_level_query = h_set_last
            elif self.use_L3:
                high_level_query = h_highest_last
            elif self.use_L2:
                high_level_query = h_rally_last
            else:
                high_level_query = h_shot_last  # fallback
            
            q = high_level_query.unsqueeze(1)
            # 使用 L1 的完整序列 (h_shot_sequence 是 PACT Encoder 輸出的 (B, T, d_model))
            # 注意: h_shot_sequence 只包含 PACT 的特徵, 但 L1 L2 L3 也是基於此, 足以表示序列時序資訊
            refined_shot_summary, td_attn_w = self.td_refinement(
                high_level_query=q,
                low_level_sequence=h_shot_sequence,
                low_level_mask=shot_mask
            )
            # eval 模式暫存 Top-Down attention weights 供視覺化
            if not self.training:
                self._last_td_attn_weights = td_attn_w.detach()
            # 殘差增強機制: 與原本的 h_shot_last 相加，而非直接取代
            # 藉由 self.td_gate 讓模型自主學習要吸取多少 TDCA 萃取出的新知識
            h_shot_last = h_shot_last + self.td_gate * refined_shot_summary

        # ===== 組合階層特徵 =====
        # 3層: [h_shot, h_rally, h_highest]
        # 4層: [h_shot, h_rally, h_game, h_set]
        hierarchy_features.append(h_shot_last)
        if self.use_L2:
            hierarchy_features.append(h_rally_last)
        if self.use_L3 and not self.use_L4:
            hierarchy_features.append(h_highest_last)
        if self.use_L4:
            hierarchy_features.append(h_game_last)
            hierarchy_features.append(h_set_last)

        # --- Temporal-Scale Adaptive Gating (TSAG) ---
        tsag_weights = None
        if self.use_temporal_scale_gating and len(hierarchy_features) > 1:
            lengths_clamped = shot_seq_current_lengths.clamp(max=255)
            tsag_emb = self.tsag_length_embedding(lengths_clamped)  # (B, d_model)
            
            # Content-Aware: 提取 L1 摘要特徵
            l1_context = hierarchy_features[0]  # (B, d_model)
            
            # 拼接: Length Embedding + L1 Context
            gate_input = torch.cat([tsag_emb, l1_context], dim=-1)  # (B, 2 * d_model)
            
            tsag_logits = self.tsag_gate(gate_input)                # (B, num_tasks * num_levels)

            # 截斷到實際階層數量 (相容 L1_L2 等模式)
            actual_levels = len(hierarchy_features)
            
            # Reshape 為 (B, num_tasks, num_levels)
            tsag_logits = tsag_logits.view(B_main, self.tsag_num_tasks, self.tsag_num_levels)
            tsag_logits = tsag_logits[:, :, :actual_levels]  # 截斷 (B, num_tasks, actual_levels)
            
            # 修正: 放棄 Softmax (會導致特徵能量塌陷 1/N)
            # 改用 Sigmoid * 2，初始化為 0 時輸出剛好是 1.0，完美保持特徵原本尺度
            tsag_weights = torch.sigmoid(tsag_logits) * 2.0  # (B, num_tasks, actual_levels)
            # 將權重傳遞到 fusion_module 中，每個任務會在 Cross-Attn 前套用自己的權重

        # ===== Fusion and Prediction =====
        e_player, e_opponent = self.player_encoder(player_id, opponent_id)

        # --- Turn-Based Style Gating (Hard Swap) ---
        if self.use_turn_based_gating:
            # 確定性先驗: 偶數拍代表下一拍是本方出手，奇數拍是對手
            is_next_self = (shot_seq_current_lengths % 2 == 0).unsqueeze(-1)  # (B, 1)
            
            # 物理置換 (Hard Swap)：
            # e_hitter: 下一拍負責出手的進攻方
            e_hitter = torch.where(is_next_self, e_player, e_opponent)
            # e_receiver: 下一拍負責防守的被動方
            e_receiver = torch.where(is_next_self, e_opponent, e_player)
            
            # 將 Token 的語義從 [主視角, 對手] 強制轉換為 [進攻者, 防守者]
            e_player, e_opponent = e_hitter, e_receiver

        if self.use_sequence_fusion and self.fusion_type == 'task_attention':
            # Sequence-Level Fusion: 傳遞完整拍序列給 TaskAttentionFusion
            fusion_output = self.fusion_module(
                hierarchy_features, e_player, e_opponent,
                shot_sequence=h_shot_sequence,
                shot_mask=shot_mask,
                tsag_weights=tsag_weights
            )
        else:
            fusion_output = self.fusion_module(
                hierarchy_features, e_player, e_opponent, 
                tsag_weights=tsag_weights
            )

        # Skip Connection: 將最後一拍的原始嵌入注入 Fusion 輸出
        if self.use_skip_connection:
            fusion_output = self._apply_skip_connection(fusion_output, last_shot_proj)

        logits = self.prediction_head(fusion_output)

        # === Eval 模式: 回傳內部診斷資訊供視覺化分析 ===
        if not self.training:
            debug_info = {
                # TSAG 階層權重: (B, num_tasks, num_levels) or None
                'tsag_weights': tsag_weights.detach() if tsag_weights is not None else None,
                # Top-Down Gate 學習值 (scalar)
                'td_gate_value': self.td_gate.detach().item() if self.use_top_down_attention else None,
                # Top-Down Cross-Attention weights: (B, nhead, 1, T) — 已在 TopDownRefinement 中計算
                'td_attn_weights': None,
                # 回合拍數 (用於分組統計)
                'rally_lengths': shot_seq_current_lengths.detach(),
                # Task Attention weights: {task_name: (B, nhead, 1, num_kv_tokens)}
                'task_attn_weights': None,
            }
            # 從 TopDownRefinement 提取 (已在 forward 中回傳 attn_weights)
            # 注意: attn_weights 在上方 td_refinement() 呼叫時已產生，需要暫存
            # 由於 TopDownRefinement.forward 已回傳 attn_weights，我們改在此計算時存取
            if self.use_top_down_attention and hasattr(self, '_last_td_attn_weights'):
                debug_info['td_attn_weights'] = self._last_td_attn_weights
            # 從 TaskAttentionFusion 提取
            if hasattr(self.fusion_module, 'last_attn_weights'):
                debug_info['task_attn_weights'] = self.fusion_module.last_attn_weights
            # 從 CLSTokenFusion 的內部 TaskProjectionFusion 提取
            elif hasattr(self.fusion_module, 'cls_fusion') and hasattr(self.fusion_module.cls_fusion, 'last_attn_weights'):
                debug_info['task_attn_weights'] = self.fusion_module.cls_fusion.last_attn_weights
            return logits, debug_info

        return logits
