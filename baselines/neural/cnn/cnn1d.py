from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv → activation → dropout with optional residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dropout: float,
        dilation: int,
        residual: bool,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.activation(out)
        out = self.dropout(out)
        if self.residual:
            out = out + x
        return out


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
        puzzle_emb_ndim: int = 0,
        num_puzzle_identifiers: int = 0,
        puzzle_emb_init_std: float = 0.02,
        puzzle_proj_init_std: float = 0.02,
        pad_token_id: Optional[int] = None,
        dilation_growth: int = 2,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.requires_targets = False

        puzzle_emb_ndim = int(puzzle_emb_ndim)
        num_puzzle_identifiers = int(num_puzzle_identifiers)
        puzzle_emb_init_std = float(puzzle_emb_init_std)
        puzzle_proj_init_std = float(puzzle_proj_init_std)

        self.puzzle_embedding: Optional[nn.Embedding] = None
        self.puzzle_projector: Optional[nn.Linear] = None
        if puzzle_emb_ndim > 0 and num_puzzle_identifiers > 0:
            self.puzzle_embedding = nn.Embedding(
                num_puzzle_identifiers, puzzle_emb_ndim
            )
            nn.init.trunc_normal_(self.puzzle_embedding.weight, std=puzzle_emb_init_std)
            self.puzzle_projector = nn.Linear(
                puzzle_emb_ndim,
                embedding_dim,
                bias=True,
            )
            nn.init.trunc_normal_(self.puzzle_projector.weight, std=puzzle_proj_init_std)
            if self.puzzle_projector.bias is not None:
                nn.init.zeros_(self.puzzle_projector.bias)

        self.requires_puzzle_identifiers = self.puzzle_embedding is not None

        dilation_growth = max(1, int(dilation_growth))
        use_layer_norm = bool(use_layer_norm)

        if pad_token_id is not None:
            if pad_token_id < 0 or pad_token_id >= input_vocab_size:
                raise ValueError(
                    f"pad_token_id={pad_token_id} is out of range for vocab size {input_vocab_size}."
                )
        self.pad_token_id = pad_token_id
        if pad_token_id is not None:
            self.embedding = nn.Embedding(
                input_vocab_size, embedding_dim, padding_idx=pad_token_id
            )
        else:
            self.embedding = nn.Embedding(input_vocab_size, embedding_dim)

        self.pre_layer_norm = nn.LayerNorm(embedding_dim) if use_layer_norm else None
        self.post_layer_norm = nn.LayerNorm(hidden_channels) if use_layer_norm else None

        blocks = []
        in_channels = embedding_dim
        num_layers = max(1, int(num_layers))
        for layer_idx in range(num_layers):
            dilation = dilation_growth**layer_idx
            blocks.append(
                ConvBlock(
                    in_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    dilation=dilation,
                    residual=in_channels == hidden_channels,
                )
            )
            in_channels = hidden_channels
        self.conv_stack = nn.ModuleList(blocks)
        self.output_projection = nn.Linear(hidden_channels, output_vocab_size)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        puzzle_identifiers: Optional[torch.Tensor] = None,
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
        if self.pre_layer_norm is not None:
            embedded = self.pre_layer_norm(embedded)
        if self.puzzle_embedding is not None and self.puzzle_projector is not None:
            if puzzle_identifiers is None:
                raise ValueError(
                    "CNN1DSeq2Seq expects puzzle identifiers when puzzle embeddings are enabled."
                )
            puzzle_identifiers = puzzle_identifiers.to(inputs.device)
            puzzle_vectors = self.puzzle_embedding(puzzle_identifiers)
            puzzle_bias = self.puzzle_projector(puzzle_vectors).to(embedded.dtype)
            embedded = embedded + puzzle_bias.unsqueeze(1)
        features = embedded.transpose(1, 2)
        for block in self.conv_stack:
            features = block(features)
        features = features.transpose(1, 2)
        if self.post_layer_norm is not None:
            features = self.post_layer_norm(features)
        logits = self.output_projection(features)
        return logits
