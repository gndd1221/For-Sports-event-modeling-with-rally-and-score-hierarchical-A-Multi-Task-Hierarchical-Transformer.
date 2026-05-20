import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils.base_model import BaseModel
from src.model_components import PlayerStyleEncoder


class BaselineLSTMFlat(BaseModel):
    """Flat-history LSTM baseline: flatten L3/L2/L1 history into one sequence."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        model_args = config.get('model_args', config)

        self.d_model = model_args.get('d_model', 128)
        self.num_layers = model_args.get('num_encoder_layers', 2)
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

        self.main_lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.num_layers > 1 else 0.0,
        )

        fusion_input_dim = self.d_model + (self.player_embedding_dim * 2)
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.LayerNorm(self.d_model),
        )

        self.build_task_heads(feature_dim=self.d_model)

    def _embed_features(self, raw_features):
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
        batch_flat_lengths = []

        for b in range(batch_size):
            valid_shots_list = []

            n_sets = len_sets[b].item()
            if n_sets > 0:
                for s in range(n_sets):
                    n_rallies = len_set_rallies[b, s].item()
                    if n_rallies > 0:
                        for r in range(n_rallies):
                            n_shots = len_set_shots[b, s, r].item()
                            if n_shots > 0:
                                valid_shots_list.append(set_hist[b, s, r, :n_shots])

            n_rallies_curr = len_rallies[b].item()
            if n_rallies_curr > 0:
                for r in range(n_rallies_curr):
                    n_shots = len_rally_shots[b, r].item()
                    if n_shots > 0:
                        valid_shots_list.append(rally_hist[b, r, :n_shots])

            n_curr = len_curr_shots[b].item()
            if n_curr > 0:
                valid_shots_list.append(curr_hist[b, :n_curr])

            if len(valid_shots_list) > 0:
                full_seq = torch.cat(valid_shots_list, dim=0)
            else:
                full_seq = torch.zeros(1, curr_hist.size(-1), device=device)

            batch_flat_history.append(full_seq)
            batch_flat_lengths.append(full_seq.size(0))

        padded_flat_history = torch.nn.utils.rnn.pad_sequence(batch_flat_history, batch_first=True, padding_value=0)
        flat_lengths = torch.tensor(batch_flat_lengths, dtype=torch.long, device='cpu')

        embedded_history = self._embed_features(padded_flat_history)
        projected_history = self.input_projection(embedded_history)

        packed_input = pack_padded_sequence(
            projected_history,
            flat_lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.main_lstm(packed_input)
        final_hidden = h_n[-1]

        e_player, e_opponent = self.player_encoder(player_id, opponent_id)
        fusion_input = torch.cat([final_hidden, e_player, e_opponent], dim=-1)
        c_t = self.fusion_layer(fusion_input)
        return c_t
