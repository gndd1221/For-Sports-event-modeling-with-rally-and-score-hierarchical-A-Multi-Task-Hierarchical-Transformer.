import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils.base_model import BaseModel
from src.model_components import PlayerStyleEncoder


class BaselineHLSTM(BaseModel):
    """Hierarchical LSTM baseline (L1/L2/L3) adapted to BaseModel contract."""

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

        self.shot_lstm = nn.LSTM(
            self.d_model,
            self.d_model,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.num_layers > 1 else 0.0,
        )
        self.rally_lstm = nn.LSTM(
            self.d_model,
            self.d_model,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.num_layers > 1 else 0.0,
        )
        self.set_lstm = nn.LSTM(
            self.d_model,
            self.d_model,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout_prob if self.num_layers > 1 else 0.0,
        )

        fusion_input_dim = (self.d_model * 3) + (self.player_embedding_dim * 2)
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.LayerNorm(self.d_model),
        )

        self.empty_rally_summary = nn.Parameter(torch.zeros(self.d_model))
        self.empty_set_summary = nn.Parameter(torch.zeros(self.d_model))

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

    def _run_packed_lstm(self, lstm_layer, inputs, lengths):
        lengths_cpu = lengths.cpu().to(dtype=torch.int64)
        valid_indices = torch.nonzero(lengths_cpu > 0).squeeze()

        if valid_indices.numel() == 0:
            return torch.zeros(inputs.size(0), inputs.size(1), inputs.size(2), device=inputs.device)

        if valid_indices.ndim == 0:
            valid_indices = valid_indices.unsqueeze(0)

        valid_inputs = inputs[valid_indices]
        valid_lengths = lengths_cpu[valid_indices]

        packed_input = pack_padded_sequence(valid_inputs, valid_lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = lstm_layer(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=inputs.size(1))

        final_output = torch.zeros(inputs.size(0), inputs.size(1), output.size(2), device=inputs.device)
        final_output[valid_indices] = output
        return final_output

    def _get_last_valid_output(self, sequence_output, lengths):
        batch_size = sequence_output.size(0)
        valid_lengths = torch.clamp(lengths - 1, min=0)
        index = valid_lengths.view(batch_size, 1, 1).expand(-1, -1, sequence_output.size(-1))
        return sequence_output.gather(1, index).squeeze(1)

    def extract_features(self, batch):
        device = next(self.parameters()).device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, dict):
                for k, v in value.items():
                    batch[key][k] = v.to(device)

        player_id = batch['player_id']
        opponent_id = batch['opponent_id']
        batch_size = player_id.shape[0]

        set_history_lengths = batch['set_history_lengths']
        if set_history_lengths.numel() > 0 and set_history_lengths.max() > 0:
            set_history_raw = batch['set_history']
            bsz, num_sets, num_rallies, num_shots, num_feat = set_history_raw.shape

            set_history_embedded = self._embed_and_combine_features(set_history_raw.reshape(-1, num_feat))
            set_history_proj = self.input_projection(set_history_embedded).reshape(bsz * num_sets * num_rallies, num_shots, -1)

            set_history_shot_lengths = batch['set_history_shot_lengths'].reshape(-1)
            shot_out = self._run_packed_lstm(self.shot_lstm, set_history_proj, set_history_shot_lengths)
            rally_summaries = self._get_last_valid_output(shot_out, set_history_shot_lengths)

            set_history_rallies = rally_summaries.reshape(bsz * num_sets, num_rallies, -1)
            set_history_rally_lengths = batch['set_history_rally_lengths'].reshape(-1)
            rally_out = self._run_packed_lstm(self.rally_lstm, set_history_rallies, set_history_rally_lengths)
            set_summaries = self._get_last_valid_output(rally_out, set_history_rally_lengths)

            set_history_sets = set_summaries.reshape(bsz, num_sets, -1)
            set_out = self._run_packed_lstm(self.set_lstm, set_history_sets, set_history_lengths)
            h_set_last = self._get_last_valid_output(set_out, set_history_lengths)
        else:
            h_set_last = self.empty_set_summary.unsqueeze(0).expand(batch_size, -1)

        rally_history_lengths = batch['rally_history_lengths']
        if rally_history_lengths.numel() > 0 and rally_history_lengths.max() > 0:
            rally_history_raw = batch['rally_history']
            bsz, num_rallies, num_shots, num_feat = rally_history_raw.shape

            rally_history_embedded = self._embed_and_combine_features(rally_history_raw.reshape(-1, num_feat))
            rally_history_proj = self.input_projection(rally_history_embedded).reshape(bsz * num_rallies, num_shots, -1)

            rally_history_shot_lengths = batch['rally_history_shot_lengths'].reshape(-1)
            shot_out_r = self._run_packed_lstm(self.shot_lstm, rally_history_proj, rally_history_shot_lengths)
            rally_summaries = self._get_last_valid_output(shot_out_r, rally_history_shot_lengths)

            rally_history_rallies = rally_summaries.reshape(bsz, num_rallies, -1)
            rally_out_r = self._run_packed_lstm(self.rally_lstm, rally_history_rallies, rally_history_lengths)
            h_rally_last = self._get_last_valid_output(rally_out_r, rally_history_lengths)
        else:
            h_rally_last = self.empty_rally_summary.unsqueeze(0).expand(batch_size, -1)

        set_mask_empty = (set_history_lengths == 0).unsqueeze(1).expand_as(h_set_last)
        h_set_last = torch.where(set_mask_empty, self.empty_set_summary, h_set_last)

        rally_mask_empty = (rally_history_lengths == 0).unsqueeze(1).expand_as(h_rally_last)
        h_rally_last = torch.where(rally_mask_empty, self.empty_rally_summary, h_rally_last)

        shot_seq_current_raw = batch['shot_seq_current']
        shot_seq_current_embedded = self._embed_and_combine_features(shot_seq_current_raw)
        shot_seq_current_proj = self.input_projection(shot_seq_current_embedded)
        shot_seq_current_lengths = batch['shot_seq_current_lengths']

        shot_out_c = self._run_packed_lstm(self.shot_lstm, shot_seq_current_proj, shot_seq_current_lengths)
        h_shot_last = self._get_last_valid_output(shot_out_c, shot_seq_current_lengths)

        e_player, e_opponent = self.player_encoder(player_id, opponent_id)
        fusion_input = torch.cat([h_shot_last, h_rally_last, h_set_last, e_player, e_opponent], dim=-1)
        c_t = self.fusion_layer(fusion_input)

        return c_t
