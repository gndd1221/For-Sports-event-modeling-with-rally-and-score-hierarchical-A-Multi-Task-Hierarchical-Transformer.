import torch
import torch.nn as nn

from utils.base_model import BaseModel
from src.model_components import PlayerStyleEncoder


class BaselineITransformerFlat(BaseModel):
    """Flat-history iTransformer baseline over a single shot event sequence."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        model_args = config.get('model_args', config)

        self.d_model = model_args.get('d_model', 128)
        self.nhead = model_args.get('nhead', 4)
        self.num_layers = model_args.get('num_encoder_layers', 2)
        self.dim_feedforward = model_args.get('dim_feedforward', self.d_model * 4)
        self.dropout_prob = model_args.get('dropout', 0.1)
        self.player_embedding_dim = model_args.get('player_embedding_dim', self.d_model)
        self.max_flat_seq_len = model_args.get('max_flat_seq_len') or self._default_flat_seq_len(config)

        self.feature_indices = {name: i for i, name in enumerate(config['features_to_extract'])}
        self.player_encoder = PlayerStyleEncoder(config['num_players'], self.player_embedding_dim)

        self.embedding_layers = nn.ModuleDict()
        other_embedding_dim = 0
        for feat_name in config['categorical_features']:
            vocab_size = config.get('vocab_sizes', {}).get(feat_name, config.get(f'num_{feat_name}', 2))
            embedding_dim = config.get('embedding_dims', {}).get(feat_name, 8)
            self.embedding_layers[f'{feat_name}_embedding'] = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=0
            )
            other_embedding_dim += embedding_dim

        self.combined_feature_dim = (
            (self.player_embedding_dim * 2)
            + other_embedding_dim
            + len(config['numerical_features'])
        )

        self.temporal_embedding = nn.Linear(self.max_flat_seq_len, self.d_model)
        self.variable_norm = nn.LayerNorm(self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_prob,
            batch_first=True,
            norm_first=True,
        )
        self.variable_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.output_norm = nn.LayerNorm(self.d_model)

        fusion_input_dim = self.d_model + (self.player_embedding_dim * 2)
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.LayerNorm(self.d_model),
        )

        self.build_task_heads(feature_dim=self.d_model)

    @staticmethod
    def _default_flat_seq_len(config):
        shot_window = config.get('max_shot_seq_len', 1)
        rally_window = config.get('max_rally_seq_len', 1)
        set_window = config.get('max_set_seq_len', 0)
        game_window = config.get('max_game_seq_len', 1)
        hierarchy_levels = config.get('hierarchy_levels', ['L1', 'L2', 'L3'])

        historical_units = set_window
        if 'L4' in hierarchy_levels:
            historical_units *= game_window

        return max(shot_window * ((historical_units * rally_window) + rally_window + 1), 1)

    def _embed_features(self, raw_features):
        all_parts = []

        player_id_col = self.feature_indices['player_id']
        opponent_id_col = self.feature_indices['opponent_id']
        player_ids = raw_features[..., player_id_col].long()
        opponent_ids = raw_features[..., opponent_id_col].long()

        all_parts.append(self.player_encoder.embedding(player_ids))
        all_parts.append(self.player_encoder.embedding(opponent_ids))

        for feat_name in self.config['categorical_features']:
            col_idx = self.feature_indices[feat_name]
            feat_values = raw_features[..., col_idx].long()
            all_parts.append(self.embedding_layers[f'{feat_name}_embedding'](feat_values))

        num_feat_indices = [self.feature_indices[name] for name in self.config['numerical_features']]
        if num_feat_indices:
            all_parts.append(raw_features[..., num_feat_indices].float())

        return torch.cat(all_parts, dim=-1)

    def _flatten_history(self, batch):
        player_id = batch['player_id']
        batch_size = player_id.size(0)

        set_hist = batch['set_history']
        rally_hist = batch['rally_history']
        curr_hist = batch['shot_seq_current']
        len_sets = batch['set_history_lengths']
        len_set_rallies = batch['set_history_rally_lengths']
        len_set_shots = batch['set_history_shot_lengths']
        len_rallies = batch['rally_history_lengths']
        len_rally_shots = batch['rally_history_shot_lengths']
        len_curr_shots = batch['shot_seq_current_lengths']

        batch_flat_history = []
        for b in range(batch_size):
            valid_shots = []

            for s in range(len_sets[b].item()):
                for r in range(len_set_rallies[b, s].item()):
                    n_shots = len_set_shots[b, s, r].item()
                    if n_shots > 0:
                        valid_shots.append(set_hist[b, s, r, :n_shots])

            for r in range(len_rallies[b].item()):
                n_shots = len_rally_shots[b, r].item()
                if n_shots > 0:
                    valid_shots.append(rally_hist[b, r, :n_shots])

            n_curr = len_curr_shots[b].item()
            if n_curr > 0:
                valid_shots.append(curr_hist[b, :n_curr])

            if valid_shots:
                full_seq = torch.cat(valid_shots, dim=0)
            else:
                full_seq = torch.zeros(1, curr_hist.size(-1), device=curr_hist.device)

            batch_flat_history.append(full_seq[-self.max_flat_seq_len:])

        return torch.nn.utils.rnn.pad_sequence(
            batch_flat_history, batch_first=True, padding_value=0
        )

    def _encode_flat_history(self, embedded_history):
        batch_size, batch_seq_len, feature_dim = embedded_history.shape
        x_padded = embedded_history.new_zeros(
            batch_size, self.max_flat_seq_len, feature_dim
        )
        copy_len = min(batch_seq_len, self.max_flat_seq_len)
        x_padded[:, :copy_len, :] = embedded_history[:, -copy_len:, :]

        variable_tokens = x_padded.permute(0, 2, 1)
        variable_tokens = self.temporal_embedding(variable_tokens)
        variable_tokens = self.variable_norm(variable_tokens)
        encoded_tokens = self.variable_encoder(variable_tokens)
        return self.output_norm(torch.mean(encoded_tokens, dim=1))

    def extract_features(self, batch):
        device = next(self.parameters()).device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, dict):
                for k, v in value.items():
                    batch[key][k] = v.to(device)

        # Baselines use a shared 3L flat history view for tennis as well.
        self._downgrade_4L_to_3L(batch)

        player_id = batch['player_id']
        opponent_id = batch['opponent_id']
        flat_history = self._flatten_history(batch)
        embedded_history = self._embed_features(flat_history)
        history_context = self._encode_flat_history(embedded_history)

        e_player, e_opponent = self.player_encoder(player_id, opponent_id)
        fusion_input = torch.cat([history_context, e_player, e_opponent], dim=-1)
        return self.fusion_layer(fusion_input)
