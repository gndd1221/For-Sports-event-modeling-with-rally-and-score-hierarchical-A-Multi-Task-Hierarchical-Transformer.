import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils.base_model import BaseModel


PAD = 0


def get_pad_mask(seq):
    """(B, SeqLen) -> (B, 1, SeqLen), True = valid position."""
    return (seq != PAD).unsqueeze(-2)


# =============================================================================
# Attention Submodules (faithful to original ShuttleNet)
# =============================================================================

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            if mask.dim() == 3 and attn.dim() == 4:
                mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class TypeAreaScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q_a, k_a, v_a, q_s, k_s, v_s, mask=None):
        a2a = torch.matmul(q_a, k_a.transpose(2, 3))
        a2s = torch.matmul(q_a, k_s.transpose(2, 3))
        s2a = torch.matmul(q_s, k_a.transpose(2, 3))
        s2s = torch.matmul(q_s, k_s.transpose(2, 3))
        attention_score = (a2a + a2s + s2a + s2s) / self.temperature
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        attention_score = self.dropout(F.softmax(attention_score, dim=-1))
        output = torch.matmul(attention_score, (v_a + v_s))
        return output, attention_score


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k).transpose(1, 2)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k).transpose(1, 2)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v).transpose(1, 2)
        q, attn = self.attention(q, k, v, mask=mask)
        q = q.permute(0, 2, 1, 3).contiguous().reshape(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q


class TypeAreaMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.w_qa = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ka = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_va = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        scaling_factor = (4 * d_k) ** 0.5
        self.attention = TypeAreaScaledDotProductAttention(temperature=scaling_factor, attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q_a, k_a, v_a, q_s, k_s, v_s, mask=None):
        sz_b, len_q = q_a.size(0), q_a.size(1)
        len_k = k_a.size(1)
        len_v = v_a.size(1)
        residual_a = q_a
        residual_s = q_s
        q_a = self.w_qa(q_a).view(sz_b, len_q, self.n_head, self.d_k).transpose(1, 2)
        k_a = self.w_ka(k_a).view(sz_b, len_k, self.n_head, self.d_k).transpose(1, 2)
        v_a = self.w_va(v_a).view(sz_b, len_v, self.n_head, self.d_v).transpose(1, 2)
        q_s = self.w_qs(q_s).view(sz_b, len_q, self.n_head, self.d_k).transpose(1, 2)
        k_s = self.w_ks(k_s).view(sz_b, len_k, self.n_head, self.d_k).transpose(1, 2)
        v_s = self.w_vs(v_s).view(sz_b, len_v, self.n_head, self.d_v).transpose(1, 2)
        output, attn = self.attention(q_a, k_a, v_a, q_s, k_s, v_s, mask=mask)
        output = output.permute(0, 2, 1, 3).contiguous().reshape(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += (residual_a + residual_s)
        output = self.layer_norm(output)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


# =============================================================================
# Encoder / Decoder / Fusion Layers
# =============================================================================

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.disentangled_attention = TypeAreaMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, encode_area, encode_shot, slf_attn_mask=None):
        encode_output = self.disentangled_attention(
            encode_area, encode_area, encode_area,
            encode_shot, encode_shot, encode_shot,
            mask=slf_attn_mask,
        )
        encode_output = self.pos_ffn(encode_output)
        return encode_output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.decoder_attention = TypeAreaMultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.decoder_encoder_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, decode_area, decode_shot, encode_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        decode_output = self.decoder_attention(
            decode_area, decode_area, decode_area,
            decode_shot, decode_shot, decode_shot,
            mask=slf_attn_mask,
        )
        decode_output = self.decoder_encoder_attention(
            decode_output, encode_output, encode_output,
            mask=dec_enc_attn_mask,
        )
        decode_output = self.pos_ffn(decode_output)
        return decode_output


class GatedFusionLayer(nn.Module):
    """Position-aware gated fusion (original ShuttleNet design).
    Adapted for single-step decoder: uses step_idx to select position weights."""
    def __init__(self, d, d_inner, max_seq_len=200):
        super().__init__()
        n = 3
        self.hidden1 = nn.Linear(d, d_inner, bias=False)
        self.hidden2 = nn.Linear(d, d_inner, bias=False)
        self.hidden3 = nn.Linear(d, d_inner, bias=False)
        self.gated1 = nn.Linear(d_inner * n, d, bias=False)
        self.gated2 = nn.Linear(d_inner * n, d, bias=False)
        self.gated3 = nn.Linear(d_inner * n, d, bias=False)
        self.w_A = nn.Parameter(torch.zeros([max_seq_len, d]), requires_grad=True)
        self.w_B = nn.Parameter(torch.zeros([max_seq_len, d]), requires_grad=True)
        self.w_L = nn.Parameter(torch.zeros([max_seq_len, d]), requires_grad=True)
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x_A, x_B, x_L, step_idx):
        """
        x_A, x_B, x_L: (B, 1, D) single-step decoder outputs
        step_idx: (B,) position index for position weight selection
        """
        B, _, D = x_A.shape
        idx = torch.clamp(step_idx.long(), max=self.w_A.size(0) - 1).view(B, 1, 1).expand(-1, 1, D)
        w_A_b = self.w_A.unsqueeze(0).expand(B, -1, -1).gather(1, idx)
        w_B_b = self.w_B.unsqueeze(0).expand(B, -1, -1).gather(1, idx)
        w_L_b = self.w_L.unsqueeze(0).expand(B, -1, -1).gather(1, idx)

        h_A = self.tanh_f(self.hidden1(x_A))
        h_B = self.tanh_f(self.hidden2(x_B))
        h_L = self.tanh_f(self.hidden3(x_L))

        x_concat = torch.cat((x_A, x_B, x_L), dim=-1)
        z1 = self.sigmoid_f(self.gated1(x_concat)) * h_A
        z2 = self.sigmoid_f(self.gated2(x_concat)) * h_B
        z3 = self.sigmoid_f(self.gated3(x_concat)) * h_L

        z1 = w_A_b * z1
        z2 = w_B_b * z2
        z3 = w_L_b * z3
        return self.sigmoid_f(z1 + z2 + z3)


# =============================================================================
# Positional Encoding
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(max_len, d_model))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x, positions=None):
        """
        x: (B, T, d_model)
        positions: optional (B, T) tensor of absolute position indices.
                   If None, uses sequential [0, 1, ..., T-1].
        """
        B, T, D = x.size()
        if positions is None:
            return x + self.pos_table[:, :T].clone().detach()
        positions = positions.long().unsqueeze(-1).expand(-1, -1, D)
        positions = torch.clamp(positions, min=0, max=self.pos_table.size(1) - 1)
        pe = self.pos_table.expand(B, -1, -1).gather(1, positions)
        return x + pe


# =============================================================================
# ShuttleNet Backbone (Seq2Seq: Encoder=full seq, Decoder=single-step query)
# =============================================================================

class ShuttleNetBackbone(nn.Module):
    """
    Faithful adaptation of ShuttleNet for next-shot classification.

    Architecture:
    - Encoder (Local/TRE + Global/TPE): processes full shot_seq_current
    - Decoder (Local + Global): single-step query (last shot) cross-attends to encoder
    - GatedFusion: position-aware fusion of Local, Global-A, Global-B decoder outputs
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_args = config.get('model_args', config)

        self.d_model = model_args.get('d_model', 128)
        self.nhead = model_args.get('nhead', 4)
        self.num_layers = 1  # ShuttleNet: single-layer encoder/decoder
        self.dim_feedforward = model_args.get('dim_feedforward', self.d_model * 2)
        self.dropout_prob = model_args.get('dropout', 0.1)

        self.feature_indices = {name: i for i, name in enumerate(config['features_to_extract'])}

        # --- Embeddings ---
        self.embedding_layers = nn.ModuleDict()
        cat_emb_dim_sum = 0
        for feat_name in config['categorical_features']:
            vocab_size = config['vocab_sizes'][feat_name]
            emb_dim = config['embedding_dims'][feat_name]
            self.embedding_layers[f'{feat_name}_embedding'] = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            cat_emb_dim_sum += emb_dim

        self.num_numerical = len(config['numerical_features'])
        self.type_projection = nn.Linear(cat_emb_dim_sum, self.d_model)
        self.area_projection = nn.Linear(self.num_numerical, self.d_model)

        # 與原始 ShuttleNet 一致：player embedding 直接為 d_model 維度
        self.player_embedding = nn.Embedding(config['num_players'], self.d_model, padding_idx=0)

        self.primary_shot_feature = 'type' if 'type' in self.feature_indices else config['categorical_features'][0]

        # --- PE & Dropout ---
        max_len = config.get('max_shot_seq_len', 200)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=max_len * 2)
        self.dropout = nn.Dropout(self.dropout_prob)

        d_k = self.d_model // self.nhead
        d_v = self.d_model // self.nhead

        # --- Encoder: Local + Global (shared A/B, single layer) ---
        self.local_encoder = EncoderLayer(self.d_model, self.dim_feedforward, self.nhead, d_k, d_v, self.dropout_prob)
        self.global_encoder = EncoderLayer(self.d_model, self.dim_feedforward, self.nhead, d_k, d_v, self.dropout_prob)

        # --- Decoder: Local + Global (shared A/B, single layer) ---
        self.local_decoder = DecoderLayer(self.d_model, self.dim_feedforward, self.nhead, d_k, d_v, self.dropout_prob)
        self.global_decoder = DecoderLayer(self.d_model, self.dim_feedforward, self.nhead, d_k, d_v, self.dropout_prob)

        # --- Gated Fusion ---
        self.fusion_layer = GatedFusionLayer(self.d_model, self.d_model, max_seq_len=max_len)

    def _get_separated_embeddings(self, raw_features):
        cat_parts = []
        for feat_name in self.config['categorical_features']:
            col_idx = self.feature_indices[feat_name]
            feat_values = raw_features[..., col_idx].long()
            cat_parts.append(self.embedding_layers[f'{feat_name}_embedding'](feat_values))
        type_feat = torch.cat(cat_parts, dim=-1)

        num_indices = [self.feature_indices[name] for name in self.config['numerical_features']]
        area_feat = raw_features[..., num_indices].float()

        player_idx = self.feature_indices['player_id']
        opponent_idx = self.feature_indices['opponent_id']
        player_ids = raw_features[..., player_idx].long()
        opponent_ids = raw_features[..., opponent_idx].long()
        return type_feat, area_feat, player_ids, opponent_ids

    def forward(self, batch):
        device = next(self.parameters()).device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, dict):
                for k, v in value.items():
                    batch[key][k] = v.to(device)

        shot_seq_raw = batch['shot_seq_current']
        lengths = batch['shot_seq_current_lengths']
        B, SeqLen, _ = shot_seq_raw.shape

        # =====================================================================
        # 1. Feature Embedding (same as original ShuttleNet: type + player, area + player)
        # =====================================================================
        shot_token_col = self.feature_indices[self.primary_shot_feature]
        shot_tokens = shot_seq_raw[..., shot_token_col].long()

        type_feat, area_feat, player_ids, _ = self._get_separated_embeddings(shot_seq_raw)
        # 與原始 ShuttleNet 一致：player embedding 直接 d_model 維度，不需要 projection
        embedded_player = self.player_embedding(player_ids)

        type_proj = self.type_projection(type_feat) + embedded_player       # h_s = embedded_shot + embedded_player
        area_proj = F.relu(self.area_projection(area_feat)) + embedded_player  # h_a = F.relu(embedded_area) + embedded_player

        # =====================================================================
        # 2. Encoder: Local (TRE) + Global (TPE for A/B)
        # =====================================================================
        last_idx = torch.clamp(lengths - 1, min=0)
        batch_idx = torch.arange(B, device=device)

        enc_shot_tokens = shot_tokens.clone()
        enc_shot_tokens[batch_idx, last_idx] = PAD
        
        enc_mask_local = get_pad_mask(enc_shot_tokens)  # (B, 1, SeqLen)

        enc_area = self.dropout(self.pos_encoder(area_proj))
        enc_type = self.dropout(self.pos_encoder(type_proj))
        encode_local = self.local_encoder(enc_area, enc_type, slf_attn_mask=enc_mask_local)

        # Global: split A (::2) and B (1::2)
        tokens_a = enc_shot_tokens[:, ::2]
        tokens_b = enc_shot_tokens[:, 1::2]
        type_a, type_b = type_proj[:, ::2], type_proj[:, 1::2]
        area_a, area_b = area_proj[:, ::2], area_proj[:, 1::2]

        mask_a = get_pad_mask(tokens_a)
        enc_area_a = self.dropout(self.pos_encoder(area_a))
        enc_type_a = self.dropout(self.pos_encoder(type_a))
        encode_global_a = self.global_encoder(enc_area_a, enc_type_a, slf_attn_mask=mask_a)

        if tokens_b.size(1) > 0:
            mask_b = get_pad_mask(tokens_b)
            enc_area_b = self.dropout(self.pos_encoder(area_b))
            enc_type_b = self.dropout(self.pos_encoder(type_b))
            encode_global_b = self.global_encoder(enc_area_b, enc_type_b, slf_attn_mask=mask_b)
        else:
            encode_global_b = torch.zeros(B, 0, self.d_model, device=device)
            mask_b = None

        # =====================================================================
        # 3. Decoder: single-step query (last observed shot) → Cross-Attn to Encoder
        # =====================================================================
        # Extract last shot's type/area embeddings as decoder query: (B, 1, d_model)
        dec_type_query = type_proj[batch_idx, last_idx].unsqueeze(1)
        dec_area_query = area_proj[batch_idx, last_idx].unsqueeze(1)

        # PE at the next position (position = lengths, predicting the shot at this position)
        dec_positions = lengths.unsqueeze(1)  # (B, 1)
        dec_type_query = self.dropout(self.pos_encoder(dec_type_query, positions=dec_positions))
        dec_area_query = self.dropout(self.pos_encoder(dec_area_query, positions=dec_positions))

        # Local decoder: cross-attend to local encoder output
        decode_local = self.local_decoder(
            dec_area_query, dec_type_query, encode_local,
            slf_attn_mask=None,  # single step, no causal mask needed
            dec_enc_attn_mask=enc_mask_local,
        )

        # Global decoder A: cross-attend to global encoder A
        decode_global_a = self.global_decoder(
            dec_area_query, dec_type_query, encode_global_a,
            slf_attn_mask=None,
            dec_enc_attn_mask=mask_a,
        )

        # Global decoder B: cross-attend to global encoder B
        if encode_global_b.size(1) > 0:
            decode_global_b = self.global_decoder(
                dec_area_query, dec_type_query, encode_global_b,
                slf_attn_mask=None,
                dec_enc_attn_mask=mask_b,
            )
        else:
            decode_global_b = torch.zeros(B, 1, self.d_model, device=device)

        # =====================================================================
        # 4. Gated Fusion (position-aware, single step)
        # =====================================================================
        fused = self.fusion_layer(decode_global_a, decode_global_b, decode_local, lengths)

        # =====================================================================
        # 5. Add target player embedding → output context vector
        # =====================================================================
        final_context = fused.squeeze(1)  # (B, d_model)
        # 與原始 ShuttleNet ShotGenPredictor 一致：加上 target player embedding
        target_player_ids = batch['player_id'].long()
        target_player_emb = self.player_embedding(target_player_ids)
        final_context = final_context + target_player_emb

        return final_context


# =============================================================================
# PACTModel wrapper (integrates with BaseModel framework)
# =============================================================================

class PACTModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = ShuttleNetBackbone(config)
        self.build_task_heads(feature_dim=self.backbone.d_model)

        # Override heads to use bias=False, matching original ShuttleNet
        model_args = config.get('model_args', config)
        for task in self.task_names:
            head_key = f'head_{task}'
            if head_key in self.task_heads:
                num_classes = model_args.get(f'num_{task}', config.get(f'num_{task}', 2))
                self.task_heads[head_key] = nn.Linear(self.backbone.d_model, num_classes, bias=False)

    def extract_features(self, batch):
        return self.backbone(batch)