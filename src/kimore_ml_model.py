"""Compact TCN-BiGRU-attention regressor for multi-exercise KIMORE data."""

from __future__ import annotations

import torch
from torch import nn

from kimore_ml_data import EXERCISES
from kimore_yu_xiong_dtw import FEATURE_DIMENSIONS


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = 2 * dilation
        self.layers = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=padding,
                dilation=dilation,
            ),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()

    def forward(self, values: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        residual = values
        output = self.activation(self.layers(values) + residual)
        return output * frame_mask.unsqueeze(1)


class TemporalScoreModel(nn.Module):
    """Predict clinical TS from motion, QC masks, and exercise identity.

    The TCN extracts short- and medium-range motion patterns, the bidirectional
    GRU models the complete execution, and masked attention selects the most
    informative time steps.  A separate head is learned for every exercise.
    """

    def __init__(
        self,
        channels: int = 64,
        gru_hidden: int = 64,
        exercise_embedding_dim: int = 8,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if channels % 8:
            raise ValueError("TCN channels must be divisible by eight")
        vector_channels = FEATURE_DIMENSIONS * 3
        observation_channels = FEATURE_DIMENSIONS
        self.input_projection = nn.Sequential(
            nn.Conv1d(vector_channels + observation_channels, channels, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )
        self.temporal_blocks = nn.ModuleList(
            TemporalResidualBlock(channels, dilation, dropout)
            for dilation in (1, 2, 4)
        )
        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        encoded_size = gru_hidden * 2
        self.attention = nn.Sequential(
            nn.Linear(encoded_size, gru_hidden),
            nn.Tanh(),
            nn.Linear(gru_hidden, 1),
        )
        self.exercise_embedding = nn.Embedding(
            len(EXERCISES), exercise_embedding_dim
        )
        head_input = encoded_size + exercise_embedding_dim
        self.exercise_heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(head_input, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            for _ in EXERCISES
        )

    def forward(
        self,
        vectors: torch.Tensor,
        frame_mask: torch.Tensor,
        component_observed_mask: torch.Tensor,
        exercise_indices: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if vectors.ndim != 4 or vectors.shape[2:] != (FEATURE_DIMENSIONS, 3):
            raise ValueError("Vectors must have shape (batch, time, 9, 3)")
        if frame_mask.shape != vectors.shape[:2]:
            raise ValueError("Frame mask must match batch and time dimensions")
        if component_observed_mask.shape != vectors.shape[:3]:
            raise ValueError(
                "Component-observed mask must have shape (batch, time, 9)"
            )
        if exercise_indices.shape != (len(vectors),):
            raise ValueError("Exercise indices must contain one value per sample")
        if not torch.all(frame_mask.any(dim=1)):
            raise ValueError("Every sample must contain at least one usable frame")

        flattened_vectors = vectors.flatten(start_dim=2)
        model_input = torch.cat(
            (flattened_vectors, component_observed_mask.to(vectors.dtype)), dim=2
        )
        model_input = model_input * frame_mask.unsqueeze(2)
        encoded = self.input_projection(model_input.transpose(1, 2))
        encoded = encoded * frame_mask.unsqueeze(1)
        for block in self.temporal_blocks:
            encoded = block(encoded, frame_mask)
        encoded, _ = self.gru(encoded.transpose(1, 2))

        attention_logits = self.attention(encoded).squeeze(2)
        attention_logits = attention_logits.masked_fill(
            ~frame_mask, torch.finfo(attention_logits.dtype).min
        )
        attention_weights = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(encoded * attention_weights.unsqueeze(2), dim=1)
        exercise_context = self.exercise_embedding(exercise_indices)
        combined = torch.cat((pooled, exercise_context), dim=1)
        all_predictions = torch.cat(
            [head(combined) for head in self.exercise_heads], dim=1
        )
        predictions = all_predictions.gather(
            1, exercise_indices.unsqueeze(1)
        ).squeeze(1)
        if return_attention:
            return predictions, attention_weights
        return predictions


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
