import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from utils.base_model import BaseModel
from src.model_components import PlayerStyleEncoder


class BaselineLSTMContext(BaseModel):
    """Simple LSTM + rally history context baseline."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        model_args = config.get('model_args', config)

        self.d_model = model_args.get('d_model', 128)
        self.num_layers = model_args.get('num_encoder_layers', 1)
        self.dropout_prob = model_args.get('dropout', 0.1)
        self.player_embedding_dim = model_args.get('player_embedding_dim', self.d_model)

        self.feature_indices = {name: i for i, name in enumerate(config['features_to_extract'])}

        self.player_encoder = PlayerStyleEncoder(config['num_players'], self.player_embedding_dim)

        self.embedding_layers = nn.ModuleDict()
        other_embedding_dim = 0
        for feat_name in config['categorical_features']:
            vocab_size = config.get('vocab_sizes', {}).get(feat_name, config.get(f'num_{feat_name}', 2))
            embedding_dim = config.get('embedding_dims', {}).get(feat_name, 8)
            self.embedding_layers[f'{feat_name}_embedding'] = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            other_embedding_dim += embedding_dim

        projector_input_dim = (self.player_embedding_dim * 2) + other_embedding_dim + len(config['numerical_features'])
        self.input_projection = nn.Sequential(
            nn.Linear(projector_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(self.dropout_prob),
        )

        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.num_layers > 1 else 0.0,
        )

        fusion_input_dim = (self.d_model * 2) + (self.player_embedding_dim * 2)
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.LayerNorm(self.d_model),
        )

        self.empty_history_context = nn.Parameter(torch.zeros(self.d_model))

        self.build_task_heads(feature_dim=self.d_model)

    def _embed_and_combine_features(self, raw_features):
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
            embedded = self.embedding_layers[f'{feat_name}_embedding'](feat_values)
            all_parts.append(embedded)

        num_feat_indices = [self.feature_indices[name] for name in self.config['numerical_features']]
        if len(num_feat_indices) > 0:
            numerical_features = raw_features[..., num_feat_indices].float()
            all_parts.append(numerical_features)

        return torch.cat(all_parts, dim=-1)

    def _get_history_context(self, history_tensor, history_lengths):
        if history_lengths.numel() == 0 or history_lengths.max() == 0:
            return self.empty_history_context.expand(history_tensor.size(0), -1)

        bsz, num_rallies, num_shots, num_feat = history_tensor.shape

        history_flat = history_tensor.reshape(bsz, num_rallies * num_shots, num_feat)
        history_embedded = self._embed_and_combine_features(history_flat)
        history_proj = self.input_projection(history_embedded)

        mask = (history_flat.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
        sum_features = torch.sum(history_proj * mask, dim=1)
        sum_counts = torch.sum(mask, dim=1).clamp(min=1.0)
        mean_context = sum_features / sum_counts

        return mean_context

    def _run_lstm(self, inputs, lengths):
        lengths_cpu = lengths.cpu().to(dtype=torch.int64)
        valid_indices = torch.nonzero(lengths_cpu > 0).squeeze()

        if valid_indices.numel() == 0:
            return torch.zeros(inputs.size(0), inputs.size(2), device=inputs.device)

        if valid_indices.ndim == 0:
            valid_indices = valid_indices.unsqueeze(0)

        valid_inputs = inputs[valid_indices]
        valid_lengths = lengths_cpu[valid_indices]

        packed_input = pack_padded_sequence(valid_inputs, valid_lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed_input)
        last_hidden = h_n[-1]

        final_output = torch.zeros(inputs.size(0), inputs.size(2), device=inputs.device)
        final_output[valid_indices] = last_hidden
        return final_output

    def extract_features(self, batch):
        device = next(self.parameters()).device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, dict):
                for k, v in value.items():
                    batch[key][k] = v.to(device)

        # 4 層 (tennis) → 3 層降級
        self._downgrade_4L_to_3L(batch)

        player_id = batch['player_id']
        opponent_id = batch['opponent_id']

        shot_seq_current_raw = batch['shot_seq_current']
        shot_seq_current_lengths = batch['shot_seq_current_lengths']

        current_embedded = self._embed_and_combine_features(shot_seq_current_raw)
        current_proj = self.input_projection(current_embedded)
        h_current = self._run_lstm(current_proj, shot_seq_current_lengths)

        rally_history_raw = batch['rally_history']
        rally_history_lengths = batch['rally_history_lengths']
        h_history = self._get_history_context(rally_history_raw, rally_history_lengths)

        e_player, e_opponent = self.player_encoder(player_id, opponent_id)
        fusion_input = torch.cat([h_current, h_history, e_player, e_opponent], dim=-1)
        c_t = self.fusion_layer(fusion_input)

        return c_t
