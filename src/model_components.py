"""
model_components.py — 模型共用元件模組

Final MT-HTA 與 ablation 共用的模型元件，包含：
1. PositionalEncoding — Transformer 位置編碼
2. HierarchicalEncoder — 各階層共用的 Transformer 編碼器
3. PlayerStyleEncoder — 球員風格嵌入
4. CLSTokenFusion — CLS Token 融合模組
5. TaskProjectionFusion — 任務專屬投影融合
6. TaskAttentionFusion — 任務專屬 Cross-Attention 融合 (對應 task_attention)
7. DirectPredictionHead — 直接線性預測頭
8. ProjectedPredictionHead — 投影式預測頭
"""

import torch
import torch.nn as nn
import math


# =====================================================================================
# 1. 位置編碼 (Positional Encoding)
# =====================================================================================
class PositionalEncoding(nn.Module):
    """為 Transformer 注入序列的位置資訊"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



# =====================================================================================
# 2. 階層式編碼器 (統一 ShotEncoder / RallyEncoder / SetEncoder)
#    三者結構完全相同，因此合併為單一類別。
# =====================================================================================
class HierarchicalEncoder(nn.Module):
    """
    通用的階層式 Transformer 編碼器。
    可用於 L1 (Shot)、L2 (Rally)、L3 (Set) 任意層級。
    """
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, sequence, src_key_padding_mask=None, lengths=None):
        """
        Args:
            sequence: (B, S, d_model) — 輸入序列
            src_key_padding_mask: (B, S) — True 表示 padding 位置
            lengths: (B,) 或 (B, 1) — 每筆資料的有效長度
        Returns:
            (B, S, d_model) — 編碼後的序列
        """
        final_output = torch.zeros_like(sequence)
        valid_indices = (lengths > 0).view(-1)
        if not valid_indices.any():
            return final_output
        
        valid_inputs = sequence[valid_indices]
        valid_masks = src_key_padding_mask[valid_indices] if src_key_padding_mask is not None else None
        
        # Positional Encoding 需要 (S, B, E) 格式
        x = valid_inputs.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # 回到 (B, S, E)
        
        h_sequence = self.transformer_encoder(x, src_key_padding_mask=valid_masks)
        final_output[valid_indices] = h_sequence
        return final_output


# =====================================================================================
# 3. 球員風格編碼器 (Player Style Encoder)
# =====================================================================================
class PlayerStyleEncoder(nn.Module):
    def __init__(self, num_players, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_players, embedding_dim, padding_idx=0)

    def forward(self, player_id, opponent_id):
        e_player = self.embedding(player_id)
        e_opponent = self.embedding(opponent_id)
        return e_player, e_opponent


# =====================================================================================
# 4. CLS Token 融合模組
#    動態接受不同數量的階層特徵
# =====================================================================================
# =====================================================================================
# Feature-token iTransformer Encoder
# =====================================================================================
class FeatureTokenITransformerEncoder(nn.Module):
    """
    iTransformer-style encoder that treats each input feature as one token.

    The temporal axis is projected inside each feature token with right alignment
    and an explicit valid mask, so DataLoader padding never changes.
    """
    def __init__(self, config, d_model, nhead, num_layers, dim_feedforward, dropout, max_seq_len):
        super().__init__()
        self.config = config
        self.d_model = d_model
        self.max_seq_len = max(max_seq_len, 1)
        self.feature_indices = {name: i for i, name in enumerate(config['features_to_extract'])}
        self.categorical_features = [
            name for name in config['categorical_features']
            if name not in {'player_id', 'opponent_id'}
        ]
        self.numerical_features = [
            name for name in config['numerical_features']
            if name not in {'player_id', 'opponent_id'}
        ]
        self.feature_names = self.categorical_features + self.numerical_features
        self.num_categorical = len(self.categorical_features)

        self.categorical_embeddings = nn.ModuleDict()
        self.categorical_projections = nn.ModuleDict()
        for feat_name in self.categorical_features:
            safe_key = self._module_key(feat_name)
            vocab_size = config.get('vocab_sizes', {}).get(feat_name, config.get(f'num_{feat_name}', 2))
            embedding_dim = config.get('embedding_dims', {}).get(feat_name, 8)
            self.categorical_embeddings[safe_key] = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=0
            )
            self.categorical_projections[safe_key] = nn.Linear(embedding_dim, d_model)

        self.numerical_projections = nn.ModuleDict({
            self._module_key(feat_name): nn.Linear(1, d_model)
            for feat_name in self.numerical_features
        })
        self.temporal_projections = nn.ModuleDict({
            self._module_key(feat_name): nn.Linear(self.max_seq_len, 1, bias=False)
            for feat_name in self.feature_names
        })

        self.input_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.feature_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, raw_sequence, lengths):
        tokens = []
        for feat_name in self.categorical_features:
            safe_key = self._module_key(feat_name)
            col_idx = self.feature_indices[feat_name]
            values = raw_sequence[..., col_idx].long()
            embedded = self.categorical_embeddings[safe_key](values)
            projected = self.categorical_projections[safe_key](embedded)
            tokens.append(self._temporal_project(feat_name, projected, lengths))

        for feat_name in self.numerical_features:
            safe_key = self._module_key(feat_name)
            col_idx = self.feature_indices[feat_name]
            values = raw_sequence[..., col_idx].float().unsqueeze(-1)
            projected = self.numerical_projections[safe_key](values)
            tokens.append(self._temporal_project(feat_name, projected, lengths))

        return self._encode_feature_tokens(tokens, lengths)

    def summarize_shot_units(self, raw_units, shot_lengths):
        """
        Summarize each raw shot unit into feature tokens.

        Args:
            raw_units: (*, T, F)
            shot_lengths: (*)
        Returns:
            (*, num_features, d_model)
        """
        prefix_shape = raw_units.shape[:-2]
        flat_units = raw_units.reshape(-1, raw_units.size(-2), raw_units.size(-1))
        flat_lengths = shot_lengths.reshape(-1).clamp(max=flat_units.size(1))
        valid = flat_lengths > 0
        mask = self._create_time_mask(flat_lengths, flat_units.size(1)).unsqueeze(-1)
        tokens = []

        for feat_name in self.categorical_features:
            safe_key = self._module_key(feat_name)
            col_idx = self.feature_indices[feat_name]
            values = flat_units[..., col_idx].long()
            embedded = self.categorical_embeddings[safe_key](values)
            masked = embedded * mask.to(embedded.dtype)
            denom = flat_lengths.clamp(min=1).to(embedded.dtype).view(-1, 1)
            mean_emb = masked.sum(dim=1) / denom
            token = self.categorical_projections[safe_key](mean_emb)
            token = token * valid.to(token.dtype).unsqueeze(-1)
            tokens.append(token)

        last_idx = torch.clamp(flat_lengths - 1, min=0)
        batch_idx = torch.arange(flat_units.size(0), device=flat_units.device)
        for feat_name in self.numerical_features:
            safe_key = self._module_key(feat_name)
            col_idx = self.feature_indices[feat_name]
            last_values = flat_units[batch_idx, last_idx, col_idx].float().unsqueeze(-1)
            token = self.numerical_projections[safe_key](last_values)
            token = token * valid.to(token.dtype).unsqueeze(-1)
            tokens.append(token)

        stacked = torch.stack(tokens, dim=1)
        return stacked.reshape(*prefix_shape, len(self.feature_names), self.d_model)

    def aggregate_feature_units(self, unit_summaries, unit_lengths, unit_weights):
        """
        Aggregate lower-level feature summaries into one higher-level unit summary.

        Categorical features use a weighted mean; numerical/state features use
        the last valid lower-level unit.
        """
        flat = unit_summaries.reshape(
            -1, unit_summaries.size(-3), unit_summaries.size(-2), unit_summaries.size(-1)
        )
        flat_lengths = unit_lengths.reshape(-1).clamp(max=flat.size(1))
        flat_weights = unit_weights.reshape(-1, unit_weights.size(-1)).to(flat.dtype)
        valid = flat_lengths > 0
        unit_mask = self._create_time_mask(flat_lengths, flat.size(1)).to(flat.dtype)
        weights = flat_weights * unit_mask
        denom = weights.sum(dim=1).clamp(min=1.0).view(-1, 1, 1)

        parts = []
        if self.num_categorical > 0:
            cat = flat[:, :, :self.num_categorical, :]
            cat_summary = (cat * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) / denom
            parts.append(cat_summary)

        if self.num_categorical < len(self.feature_names):
            last_idx = torch.clamp(flat_lengths - 1, min=0)
            batch_idx = torch.arange(flat.size(0), device=flat.device)
            num_summary = flat[batch_idx, last_idx, self.num_categorical:, :]
            parts.append(num_summary)

        summary = torch.cat(parts, dim=1)
        summary = summary * valid.to(summary.dtype).view(-1, 1, 1)
        return summary.reshape(*unit_lengths.shape, len(self.feature_names), self.d_model)

    def encode_feature_summary_sequence(self, feature_sequence, lengths):
        """
        Args:
            feature_sequence: (B, L, num_features, d_model)
            lengths: (B,)
        Returns:
            (B, d_model)
        """
        tokens = []
        for feature_idx, feat_name in enumerate(self.feature_names):
            tokens.append(self._temporal_project(feat_name, feature_sequence[:, :, feature_idx, :], lengths))
        return self._encode_feature_tokens(tokens, lengths)

    def encode_unit_sequence(self, raw_units, unit_lengths, shot_lengths):
        unit_summaries = self.summarize_shot_units(raw_units, shot_lengths)
        return self.encode_feature_summary_sequence(unit_summaries, unit_lengths)

    def _encode_feature_tokens(self, tokens, lengths):
        if not tokens:
            raise RuntimeError("FeatureTokenITransformerEncoder requires at least one feature token.")
        feature_tokens = torch.stack(tokens, dim=1)
        feature_tokens = self.input_norm(feature_tokens)
        encoded = self.feature_encoder(feature_tokens)
        context = self.output_norm(encoded.mean(dim=1))
        valid = lengths > 0
        return context * valid.to(context.dtype).unsqueeze(-1)

    def _temporal_project(self, feat_name, sequence, lengths):
        aligned, valid_mask = self._right_align_sequence(sequence, lengths)
        aligned = aligned * valid_mask.unsqueeze(-1).to(aligned.dtype)
        projected = self.temporal_projections[self._module_key(feat_name)](aligned.transpose(1, 2)).squeeze(-1)
        return projected

    def _right_align_sequence(self, sequence, lengths):
        batch_size, seq_len, dim = sequence.shape
        aligned = sequence.new_zeros(batch_size, self.max_seq_len, dim)
        valid_mask = torch.zeros(batch_size, self.max_seq_len, device=sequence.device, dtype=torch.bool)

        for b in range(batch_size):
            valid_len = min(int(lengths[b].item()), seq_len)
            copy_len = min(valid_len, self.max_seq_len)
            if copy_len <= 0:
                continue
            src_start = valid_len - copy_len
            aligned[b, -copy_len:, :] = sequence[b, src_start:valid_len, :]
            valid_mask[b, -copy_len:] = True

        return aligned, valid_mask

    @staticmethod
    def _create_time_mask(lengths, max_len):
        return torch.arange(max_len, device=lengths.device).expand(lengths.size(0), max_len) < lengths.unsqueeze(1)

    @staticmethod
    def _module_key(feat_name):
        return f"feature_{feat_name}"


# =====================================================================================
# CLS Token Fusion
# =====================================================================================
class CLSTokenFusion(nn.Module):
    """
    使用可學習的 CLS Token 搭配 Transformer Encoder 融合多個特徵 token。
    動態支援 3 層 (shot+rally+set)、2 層 (shot+rally)、1 層 (shot) 模式。
    """
    def __init__(self, d_model, player_embedding_dim, nhead=4, dim_feedforward=256,
                 dropout=0.1, num_fusion_layers=1):
        super().__init__()
        self.d_model = d_model
        
        self.player_proj = nn.Linear(player_embedding_dim, d_model)
        self.opponent_proj = nn.Linear(player_embedding_dim, d_model)
        
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(fusion_layer, num_layers=num_fusion_layers)
        
        self.final_norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, hierarchy_features, e_player, e_opponent, tsag_weights=None):
        """
        Args:
            hierarchy_features: list of (B, d_model) tensors — [h_shot, h_rally?, h_set?]
            e_player: (B, player_dim)
            e_opponent: (B, player_dim)
            tsag_weights: Optional (B, actual_levels) — 用於調整各階層權重
        Returns:
            C_t: (B, d_model) — 融合後的上下文向量
        """
        B = hierarchy_features[0].size(0)

        e_player_proj = self.player_proj(e_player)
        e_opponent_proj = self.opponent_proj(e_opponent)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # 動態組合: [CLS, h_shot, (h_rally), (h_set), player, opponent]
        all_tokens = [cls_tokens]
        for i, h in enumerate(hierarchy_features):
            if tsag_weights is not None:
                if tsag_weights.dim() == 3:
                    # Task-Specific模式: 在 CLS Fusion 取所有任務的平均權重 (B, num_tasks, num_levels) → (B, num_levels)
                    w = tsag_weights.mean(dim=1)
                else:
                    w = tsag_weights
                if i < w.size(1):
                    h = h * w[:, i:i+1]
            all_tokens.append(h.unsqueeze(1))
        all_tokens.append(e_player_proj.unsqueeze(1))
        all_tokens.append(e_opponent_proj.unsqueeze(1))
        
        all_tokens = torch.cat(all_tokens, dim=1)

        fused_tokens = self.transformer_encoder(all_tokens)
        C_t = fused_tokens[:, 0, :]  # CLS token 作為輸出
        C_t = self.final_norm(C_t)
        
        return C_t


# =====================================================================================
# 5. 任務專屬投影融合 (Task-Specific Projection)
#    先用 CLS Token Fusion 取得 C_t，再為每個任務做專屬投影
# =====================================================================================
class TaskProjectionFusion(nn.Module):
    """
    先使用 CLS Token 融合取得 C_t，然後為每個任務施加專屬的
    投影層 (Linear → ReLU → Dropout)，實現特徵解耦。
    """
    def __init__(self, d_model, player_embedding_dim, task_names,
                 nhead=4, dim_feedforward=256, dropout=0.1, num_fusion_layers=1):
        super().__init__()
        self.task_names = task_names
        
        # 共用的 CLS Token Fusion
        self.cls_fusion = CLSTokenFusion(
            d_model, player_embedding_dim, nhead, dim_feedforward, dropout, num_fusion_layers
        )
        
        # 每個任務的專屬投影層
        self.task_projections = nn.ModuleDict({
            f"task_{name}": nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for name in self.task_names
        })

    def forward(self, hierarchy_features, e_player, e_opponent, tsag_weights=None):
        """
        Returns:
            Dict[task_name → (B, d_model)] — 每個任務投影後的特徵
        """
        C_t = self.cls_fusion(hierarchy_features, e_player, e_opponent, tsag_weights=tsag_weights)
        
        task_outputs = {}
        for name in self.task_names:
            key = f"task_{name}"
            task_outputs[name] = self.task_projections[key](C_t)
        
        return task_outputs


# =====================================================================================
# 6. 任務專屬 Cross-Attention 融合 (Task-Specific Attention, 用於 task_attention)
#    每個任務擁有專屬 Query Token，透過 Cross-Attention 從原始特徵動態提取
# =====================================================================================
class TaskAttentionFusion(nn.Module):
    """
    任務專屬融合模組：
    - 每個任務擁有獨立的可學習 Query Token
    - 使用 Cross-Attention 讓每個任務從 Key/Value 中動態 Attend
    - 輸出: Dict[task_name → (B, d_model)]
    
    Sequence-Level Fusion 修復:
    - 多層 Cross-Attention (num_fusion_layers > 1 時)
    - 序列 Token 加入 Positional Encoding
    - 序列 Token 獨立 LayerNorm
    """
    def __init__(self, d_model, player_embedding_dim, task_names,
                 nhead=4, dim_feedforward=256, dropout=0.1, num_fusion_layers=1,
                 use_task_decoder=False):
        super().__init__()
        self.d_model = d_model
        self.task_names = task_names
        self.use_task_decoder = use_task_decoder
        
        # 球員特徵投影層
        self.player_proj = nn.Linear(player_embedding_dim, d_model)
        self.opponent_proj = nn.Linear(player_embedding_dim, d_model)
        
        # 每個任務的專屬 Query Token (可學習參數)
        self.task_queries = nn.ParameterDict({
            f"task_{name}": nn.Parameter(torch.zeros(1, 1, d_model)) 
            for name in self.task_names
        })
        for name in self.task_names:
            nn.init.normal_(self.task_queries[f"task_{name}"], mean=0, std=0.02)
        
        # 可堆疊的多層 cross-attention
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
            )
            for _ in range(num_fusion_layers)
        ])
        
        # 序列 token 使用共用的 positional encoding
        self.seq_pe = PositionalEncoding(d_model, dropout=0.0, max_len=256)
        
        # 序列 token 使用獨立的 LayerNorm
        self.seq_norm = nn.LayerNorm(d_model)
        
        # 任務專屬 FFN
        self.task_ffn = nn.ModuleDict({
            f"task_{name}": nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            ) for name in self.task_names
        })
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.ModuleDict({
            f"task_{name}": nn.LayerNorm(d_model) for name in self.task_names
        })

        # --- Task Self-Attention Decoder ---
        if self.use_task_decoder:
            self.task_self_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
            )
            self.task_decoder_norm = nn.LayerNorm(d_model)
            # Gated Residual: 零初始化引導，sigmoid(0)=0.5
            self.task_decoder_gate = nn.Parameter(torch.zeros(1))

    def forward(self, hierarchy_features, e_player, e_opponent,
                shot_sequence=None, shot_mask=None, tsag_weights=None):
        """
        Args:
            hierarchy_features: list of (B, d_model) — [h_shot, h_rally?, h_set?]
            e_player: (B, player_dim)
            e_opponent: (B, player_dim)
            shot_sequence: Optional (B, T, d_model) — L1 完整拍序列
            shot_mask: Optional (B, T) — True = padding position
        Returns:
            Dict[task_name → (B, d_model)]
        """
        B = hierarchy_features[0].size(0)
        device = hierarchy_features[0].device

        e_player_proj = self.player_proj(e_player)
        e_opponent_proj = self.opponent_proj(e_opponent)
        
        # 動態組合 Key/Value tokens
        kv_tokens = []
        mask_parts = []

        if shot_sequence is not None:
            # PositionalEncoding 使用 sequence-first 格式，因此先轉置。
            shot_seq_t = shot_sequence.permute(1, 0, 2)      # (T, B, d_model)
            shot_seq_pe = self.seq_pe(shot_seq_t)             # (T, B, d_model)
            shot_seq_pe = shot_seq_pe.permute(1, 0, 2)        # (B, T, d_model)
            
            # 套用序列 token 專用的 LayerNorm。
            shot_tokens = self.seq_norm(shot_seq_pe)           # (B, T, d_model)
            
            kv_tokens.append(shot_tokens)
            mask_parts.append(shot_mask)                       # (B, T)
            
            # 跳過 hierarchy_features[0] (L1 摘要)，加入其餘階層摘要
            summary_tokens = []
            for h in hierarchy_features[1:]:
                t = h.unsqueeze(1)                             # (B, 1, d_model)
                summary_tokens.append(t)
                mask_parts.append(torch.zeros(B, 1, device=device, dtype=torch.bool))
            
            # 摘要 Token 用 norm1
            if summary_tokens:
                summary_cat = torch.cat(summary_tokens, dim=1)
                summary_cat = self.norm1(summary_cat)
                kv_tokens.append(summary_cat)
        else:
            # 原始路徑: 每個階層 1 個摘要向量
            for h in hierarchy_features:
                kv_tokens.append(h.unsqueeze(1))

        # Player/Opponent tokens (用 norm1 正規化)
        po_tokens = torch.cat([
            e_player_proj.unsqueeze(1),
            e_opponent_proj.unsqueeze(1)
        ], dim=1)
        
        if shot_sequence is not None:
            po_tokens = self.norm1(po_tokens)
            mask_parts.append(torch.zeros(B, 2, device=device, dtype=torch.bool))
        
        kv_tokens.append(po_tokens)
        
        all_tokens = torch.cat(kv_tokens, dim=1)  # (B, num_tokens, d_model)
        
        if shot_sequence is None:
            all_tokens = self.norm1(all_tokens)

        # 組合 key_padding_mask (如果使用 Sequence-Level Fusion)
        kv_mask = None
        if shot_sequence is not None:
            kv_mask = torch.cat(mask_parts, dim=1)  # (B, total_tokens)
        
        # 為每個任務執行 Cross-Attention
        task_outputs = {}
        task_attended = {}  # 收集每個任務 Cross-Attention 出來的結果
        collect_attn = not self.training  # eval 模式才收集 attention weights
        if collect_attn:
            self.last_attn_weights = {}  # {task_name: (B, nhead, 1, num_kv_tokens)}
        for task_idx, name in enumerate(self.task_names):
            key = f"task_{name}"
            
            # 為每個任務建立專屬的 temporal-scale 權重。
            if tsag_weights is not None and tsag_weights.dim() == 3:
                # tsag_weights: (B, num_tasks, num_levels)
                task_w = tsag_weights[:, task_idx, :]  # (B, num_levels)
                task_kv = self._build_scaled_tokens(all_tokens, task_w, shot_sequence)
            elif tsag_weights is not None and tsag_weights.dim() == 2:
                # 二維輸入代表所有任務共用同一組階層權重。
                task_kv = self._build_scaled_tokens(all_tokens, tsag_weights, shot_sequence)
            else:
                task_kv = all_tokens
            
            query = self.task_queries[key].expand(B, -1, -1)  # (B, 1, d_model)
            
            # 依序套用多層 cross-attention。
            last_attn_w = None
            for ca_layer in self.cross_attention_layers:
                attended, attn_w = ca_layer(
                    query=query, key=task_kv, value=task_kv,
                    key_padding_mask=kv_mask,
                    need_weights=collect_attn,
                    average_attn_weights=False  # 保留 per-head 資訊
                )  # (B, 1, d_model)
                query = attended  # 前一層輸出 → 下一層 Query
                if collect_attn:
                    last_attn_w = attn_w  # 只保留最後一層
            if collect_attn and last_attn_w is not None:
                self.last_attn_weights[name] = last_attn_w.detach()
            
            attended = attended.squeeze(1)  # (B, d_model)
            task_attended[name] = attended

        # Task self-attention decoder 用於任務間資訊交換。
        if self.use_task_decoder:
            # 將各任務的特徵堆疊起來: (B, num_tasks, d_model)
            stacked = torch.stack([task_attended[n] for n in self.task_names], dim=1)
            stacked_norm = self.task_decoder_norm(stacked)
            
            # 任務間 Self-Attention
            refined, _ = self.task_self_attn(
                query=stacked_norm, key=stacked_norm, value=stacked_norm
            )
            
            # Gated Residual 加回 (移除 sigmoid, 讓初始值 = 0.0)
            gate = self.task_decoder_gate
            stacked = stacked + gate * refined
            
            # 拆解回字典
            for i, name in enumerate(self.task_names):
                task_attended[name] = stacked[:, i, :]

        # 每個任務進行自己的 FFN 與 LayerNorm
        for name in self.task_names:
            key = f"task_{name}"
            attended = task_attended[name]
            
            # FFN + Residual + LayerNorm
            ffn_out = self.task_ffn[key](attended)
            task_output = self.norm2[key](attended + ffn_out)
            
            task_outputs[name] = task_output
        
        return task_outputs

    def _build_scaled_tokens(self, all_tokens, weights_1d, shot_sequence):
        """
        Args:
            all_tokens: (B, total_tokens, d_model) — 已正規化的 KV tokens
            weights_1d: (B, num_levels) — 單一任務的階層權重
            shot_sequence: Optional — 用於判斷 task_attention vs sequence_attention 模式
        Returns:
            scaled_tokens: (B, total_tokens, d_model)
        """
        B = all_tokens.size(0)
        total_tokens = all_tokens.size(1)
        actual_levels = weights_1d.size(1)
        device = all_tokens.device

        if shot_sequence is None:
            level_weights = weights_1d.unsqueeze(-1)  # (B, actual_levels, 1)
            ones_rest = torch.ones(B, total_tokens - actual_levels, 1, device=device)
            scale = torch.cat([level_weights, ones_rest], dim=1)
        else:
            T = shot_sequence.size(1)
            w_L1 = weights_1d[:, 0:1].unsqueeze(-1).expand(-1, T, -1)
            parts = [w_L1]
            for i in range(1, actual_levels):
                parts.append(weights_1d[:, i:i+1].unsqueeze(-1))
            remaining = total_tokens - T - (actual_levels - 1)
            if remaining > 0:
                parts.append(torch.ones(B, remaining, 1, device=device))
            scale = torch.cat(parts, dim=1)

        return all_tokens * scale


# =====================================================================================
# 7. Attention Pooling (可學習的序列聚合)
#    用一個 learnable query 做 cross-attention 取代 last-token / mean pooling
# =====================================================================================
class AttentionPooling(nn.Module):
    """
    使用可學習的 query 向量對序列做 cross-attention，
    產生一個固定長度的向量摘要。
    比 last-token 或 mean-pooling 更能選擇性地聚焦重要時間步。
    """
    def __init__(self, d_model, nhead=1, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, sequence, padding_mask=None):
        """
        Args:
            sequence: (B, S, d_model)
            padding_mask: (B, S) — True 表示 padding 位置
        Returns:
            (B, d_model) — 聚合後的向量
        """
        B = sequence.size(0)
        q = self.query.expand(B, -1, -1)          # (B, 1, d_model)
        
        # 防止全 padding 的 row 造成 softmax(all -inf) = NaN
        # 對 lengths=0 的 entries，暫時開啟 position 0 讓 attention 有值可算
        safe_mask = padding_mask
        if padding_mask is not None:
            all_masked = padding_mask.all(dim=-1)  # (B,) True = 該 entry 全是 padding
            if all_masked.any():
                safe_mask = padding_mask.clone()
                safe_mask[all_masked, 0] = False   # 暫時開放 position 0
        
        out, attn_w = self.attn(
            query=q, key=sequence, value=sequence,
            key_padding_mask=safe_mask,
            need_weights=not self.training,
            average_attn_weights=False
        )  # (B, 1, d_model)
        
        # eval 模式: 存 attention weights 供視覺化
        if not self.training:
            self.last_attn_weights = attn_w.detach()  # (B, nhead, 1, S)
        
        result = self.norm(out.squeeze(1))         # (B, d_model)
        
        # 對全 padding 的 entries 歸零（不應包含有意義的資訊）
        if padding_mask is not None and all_masked.any():
            result[all_masked] = 0.0
        
        return result


# =====================================================================================
# 8. 動態預測頭 (支援任意運動的任意任務組合)
#    接受單一 C_t 或 task_features dict，從 config 動態建構預測頭
#    支援 head_depth 控制 MLP 深度
# =====================================================================================
class DynamicPredictionHead(nn.Module):
    """
    動態多任務預測頭。
    從 task_configs dict 動態建立各任務的預測頭。

    Args:
        d_model: 輸入特徵維度
        task_configs: Dict[task_name → num_classes]
        head_depth: 預測頭深度 (1=單層Linear, 2+=MLP)
        dropout: MLP 中的 Dropout 率 (head_depth >= 2 時使用)
    """
    def __init__(self, d_model, task_configs, head_depth=1, dropout=0.1):
        super().__init__()
        self.task_names = list(task_configs.keys())
        self.heads = nn.ModuleDict()

        for name, num_classes in task_configs.items():
            if head_depth <= 1:
                # 原始行為：單層 Linear
                self.heads[f'head_{name}'] = nn.Linear(d_model, num_classes)
            else:
                # MLP: [Linear → GELU → Dropout] × (depth-1) → Linear
                layers = []
                for _ in range(head_depth - 1):
                    layers.extend([
                        nn.Linear(d_model, d_model),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ])
                layers.append(nn.Linear(d_model, num_classes))
                self.heads[f'head_{name}'] = nn.Sequential(*layers)

    def forward(self, features):
        """
        Args:
            features: (B, d_model) — 單一融合向量 C_t
                或 Dict[task_name → (B, d_model)] — 任務專屬特徵
        Returns:
            Dict[task_name → (B, num_classes)] — 各任務的 logits
        """
        if isinstance(features, dict):
            return {
                name: self.heads[f'head_{name}'](features[name])
                for name in self.task_names
            }
        else:
            return {
                name: self.heads[f'head_{name}'](features)
                for name in self.task_names
            }


# =====================================================================================
# 9. 由上而下的注意力提煉 (Top-Down Refinement)
#    高層（全局戰術）回頭檢視並過濾低層（完整序列）的關鍵細節
# =====================================================================================
class TopDownRefinement(nn.Module):
    """
    使用高層次摘要來提煉低層次完整序列。
    例如：用 L3 (Set) 摘要當 Query，去 L1 (Shot) 的 60 拍長序列中挑選關鍵拍。
    """
    def __init__(self, d_model, nhead=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        # 前饋網路 (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        # 序列 Normalize
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        
    def forward(self, high_level_query, low_level_sequence, low_level_mask=None):
        """
        Args:
            high_level_query: (B, 1, d_model) — 如 L3 壓縮後的摘要
            low_level_sequence: (B, T, d_model) — 如 L1 完整編碼的拍序列
            low_level_mask: (B, T) — 如 L1 的 padding mask (True 表示要遮蔽)
        Returns:
            refined_summary: (B, d_model) — 被高層意識提煉過的高純度摘要
        """
        # 注意：我們採取 pre-norm 風格
        q = self.norm_query(high_level_query)
        kv = self.norm_kv(low_level_sequence)
        
        attended, attn_weights = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=low_level_mask
        ) # attended: (B, 1, d_model)
        
        # Residual 1 (用原始 query)
        x = high_level_query + attended
        
        # FFN & Residual 2
        norm_x = self.norm_out(x)
        ffn_out = self.ffn(norm_x)
        output = x + ffn_out
        
        # 從 (B, 1, d_model) 變回 (B, d_model)
        return output.squeeze(1), attn_weights
