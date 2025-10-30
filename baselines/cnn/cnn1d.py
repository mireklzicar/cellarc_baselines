from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class CNN1DSeq2Seq(nn.Module):
    """Simple 1D convolutional baseline for sequence transduction."""

    def __init__(
        self,
        input_vocab_size: int,
        output_vocab_size: int,
        *,
        embedding_dim: int = 64,
        hidden_channels: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.1,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.requires_targets = False

        padding = kernel_size // 2
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim, padding_idx=0)

        layers = []
        in_channels = embedding_dim
        num_layers = max(1, int(num_layers))
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=padding)
            )
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_channels = hidden_channels
        self.conv_stack = nn.Sequential(*layers)
        self.output_projection = nn.Linear(hidden_channels, output_vocab_size)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs: LongTensor of shape [batch, seq_len] with token ids.
            targets: Unused but kept for a uniform signature.

        Returns:
            logits: FloatTensor of shape [batch, seq_len, output_vocab_size].
        """
        del targets  # unused
        embedded = self.embedding(inputs)  # [B, S, E]
        features = self.conv_stack(embedded.transpose(1, 2)).transpose(1, 2)
        logits = self.output_projection(features)
        return logits
