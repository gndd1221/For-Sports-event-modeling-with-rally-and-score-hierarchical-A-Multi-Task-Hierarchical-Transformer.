import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.model_components import PlayerStyleEncoder
from utils.base_model import BaseModel


class BaselinePatchTST(BaseModel):
    """Feature-wise channel-independent PatchTST adapted for classification."""

    def __init__(self, config):
        super().__init__(config)
        model_args = config.get('model_args', config)

        self.d_model = model_args.get('d_model', 128)
        self.nhead = model_args.get('nhead', 4)
        self.num_layers = model_args.get('num_encoder_layers', 2)
        self.dim_feedforward = model_args.get('dim_feedforward', self.d_model * 4)
        self.dropout_prob = model_args.get('dropout', 0.1)
        self.player_embedding_dim = model_args.get('player_embedding_dim', self.d_model)

        self.patch_len = model_args.get('patch_len', 16)
        self.patch_stride = model_args.get('patch_stride', 8)
        self.patch_feature_dim = model_args.get('patch_feature_dim', 8)
        self.patch_pooling = model_args.get('patch_pooling', 'mean')
        self.patch_max_tokens = model_args.get('patch_max_tokens', 1024)

        if self.patch_len <= 0 or self.patch_stride <= 0:
            raise ValueError('patch_len and patch_stride must be positive')
        if self.patch_pooling != 'mean':
            raise ValueError("BaselinePatchTST currently supports patch_pooling='mean' only")
        if self.d_model % self.nhead != 0:
            raise ValueError('d_model must be divisible by nhead')

        self.feature_names = list(config['features_to_extract'])
        self.feature_indices = {name: i for i, name in enumerate(self.feature_names)}
        self.categorical_features = set(config['categorical_features'])
        self.numerical_features = set(config['numerical_features'])

        self.player_encoder = PlayerStyleEncoder(
            config['num_players'], self.player_embedding_dim
        )
        self.participant_embedding = nn.Embedding(
            config['num_players'], self.patch_feature_dim, padding_idx=0
        )

        self.categorical_embeddings = nn.ModuleDict()
        self.numerical_projections = nn.ModuleDict()
        for feature_name in self.feature_names:
            if feature_name in ('player_id', 'opponent_id'):
                continue
            if feature_name in self.categorical_features:
                vocab_size = config.get('vocab_sizes', {}).get(
                    feature_name, config.get(f'num_{feature_name}', 2)
                )
                self.categorical_embeddings[
                    f'{feature_name}_embedding'
                ] = nn.Embedding(
                    vocab_size, self.patch_feature_dim, padding_idx=0
                )
            elif feature_name in self.numerical_features:
                self.numerical_projections[
                    f'{feature_name}_projection'
                ] = nn.Linear(
                    1, self.patch_feature_dim
                )
            else:
                raise ValueError(
                    f"Feature '{feature_name}' is neither categorical nor numerical"
                )

        self.patch_projection = nn.Linear(
            self.patch_len * self.patch_feature_dim, self.d_model
        )
        self.patch_norm = nn.LayerNorm(self.d_model)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.patch_max_tokens, self.d_model)
        )
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)
        self.input_dropout = nn.Dropout(self.dropout_prob)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_prob,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )
        self.patch_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        self.channel_fusion = nn.Sequential(
            nn.Linear(len(self.feature_names) * self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.LayerNorm(self.d_model),
        )
        self.context_fusion = nn.Sequential(
            nn.Linear(
                self.d_model + 2 * self.player_embedding_dim, self.d_model
            ),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.LayerNorm(self.d_model),
        )

        self.build_task_heads(feature_dim=self.d_model)

    @staticmethod
    def _move_batch_to_device(batch, device):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    batch[key][nested_key] = nested_value.to(device)

    def _flatten_history(self, batch):
        set_hist = batch['set_history']
        rally_hist = batch['rally_history']
        current_hist = batch['shot_seq_current']
        batch_size = current_hist.size(0)

        flat_histories = []
        flat_lengths = []
        for batch_idx in range(batch_size):
            history_parts = []

            num_sets = batch['set_history_lengths'][batch_idx].item()
            for set_idx in range(num_sets):
                num_rallies = batch['set_history_rally_lengths'][
                    batch_idx, set_idx
                ].item()
                for rally_idx in range(num_rallies):
                    num_shots = batch['set_history_shot_lengths'][
                        batch_idx, set_idx, rally_idx
                    ].item()
                    if num_shots > 0:
                        history_parts.append(
                            set_hist[batch_idx, set_idx, rally_idx, :num_shots]
                        )

            num_rallies = batch['rally_history_lengths'][batch_idx].item()
            for rally_idx in range(num_rallies):
                num_shots = batch['rally_history_shot_lengths'][
                    batch_idx, rally_idx
                ].item()
                if num_shots > 0:
                    history_parts.append(
                        rally_hist[batch_idx, rally_idx, :num_shots]
                    )

            num_current_shots = batch['shot_seq_current_lengths'][batch_idx].item()
            if num_current_shots > 0:
                history_parts.append(current_hist[batch_idx, :num_current_shots])

            if history_parts:
                flat_history = torch.cat(history_parts, dim=0)
            else:
                flat_history = torch.zeros(
                    1,
                    current_hist.size(-1),
                    dtype=current_hist.dtype,
                    device=current_hist.device,
                )

            flat_histories.append(flat_history)
            flat_lengths.append(flat_history.size(0))

        padded_history = nn.utils.rnn.pad_sequence(
            flat_histories, batch_first=True, padding_value=0
        )
        lengths = torch.tensor(
            flat_lengths, dtype=torch.long, device=current_hist.device
        )
        return padded_history, lengths

    def _encode_feature_channels(self, raw_history, lengths):
        _, sequence_length, _ = raw_history.shape
        valid_steps = (
            torch.arange(sequence_length, device=raw_history.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )

        channel_sequences = []
        for feature_name in self.feature_names:
            values = raw_history[..., self.feature_indices[feature_name]]
            if feature_name in ('player_id', 'opponent_id'):
                encoded = self.participant_embedding(values.long())
            elif feature_name in self.categorical_features:
                encoded = self.categorical_embeddings[
                    f'{feature_name}_embedding'
                ](values.long())
            else:
                encoded = self.numerical_projections[
                    f'{feature_name}_projection'
                ](values.float().unsqueeze(-1))
            encoded = encoded * valid_steps.unsqueeze(-1).to(encoded.dtype)
            channel_sequences.append(encoded)

        channels = torch.stack(channel_sequences, dim=1)
        return channels

    def _create_patches(self, channels, lengths):
        _, _, sequence_length, _ = channels.shape
        if sequence_length <= self.patch_len:
            num_patches = 1
        else:
            num_patches = 1 + math.ceil(
                (sequence_length - self.patch_len) / self.patch_stride
            )

        padded_length = (
            (num_patches - 1) * self.patch_stride + self.patch_len
        )
        channels = F.pad(
            channels, (0, 0, 0, padded_length - sequence_length)
        )
        patches = channels.unfold(
            dimension=2, size=self.patch_len, step=self.patch_stride
        )
        patches = patches.permute(0, 1, 2, 4, 3).contiguous()
        patches = patches.flatten(start_dim=-2)

        patch_starts = (
            torch.arange(num_patches, device=lengths.device) * self.patch_stride
        )
        valid_patches = patch_starts.unsqueeze(0) < lengths.unsqueeze(1)
        return patches, valid_patches

    def _encode_patches(self, patches, valid_patches):
        batch_size, num_channels, num_patches, _ = patches.shape
        if num_patches > self.patch_max_tokens:
            raise ValueError(
                f'Patch count {num_patches} exceeds patch_max_tokens '
                f'{self.patch_max_tokens}'
            )

        patch_tokens = self.patch_norm(self.patch_projection(patches))
        patch_tokens = patch_tokens + self.positional_encoding[:, :num_patches]
        patch_tokens = self.input_dropout(patch_tokens)
        patch_tokens = patch_tokens.reshape(
            batch_size * num_channels, num_patches, self.d_model
        )

        expanded_mask = valid_patches.unsqueeze(1).expand(
            batch_size, num_channels, num_patches
        )
        expanded_mask = expanded_mask.reshape(
            batch_size * num_channels, num_patches
        )
        encoded = self.patch_encoder(
            patch_tokens, src_key_padding_mask=~expanded_mask
        )
        encoded = encoded.reshape(
            batch_size, num_channels, num_patches, self.d_model
        )

        pooling_mask = valid_patches[:, None, :, None].to(encoded.dtype)
        pooled = (encoded * pooling_mask).sum(dim=2)
        pooled = pooled / pooling_mask.sum(dim=2).clamp_min(1.0)
        return pooled

    def extract_features(self, batch):
        device = next(self.parameters()).device
        self._move_batch_to_device(batch, device)
        self._downgrade_4L_to_3L(batch)

        raw_history, history_lengths = self._flatten_history(batch)
        channels = self._encode_feature_channels(raw_history, history_lengths)
        patches, valid_patches = self._create_patches(channels, history_lengths)
        channel_summaries = self._encode_patches(patches, valid_patches)

        temporal_context = self.channel_fusion(
            channel_summaries.flatten(start_dim=1)
        )
        player_embedding, opponent_embedding = self.player_encoder(
            batch['player_id'], batch['opponent_id']
        )
        return self.context_fusion(
            torch.cat(
                [temporal_context, player_embedding, opponent_embedding], dim=-1
            )
        )
