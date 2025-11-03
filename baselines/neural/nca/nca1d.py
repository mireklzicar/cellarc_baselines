from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn


class NCA1DSeq2Seq(nn.Module):
    """Neural cellular automata sequence baseline inspired by the JAX notebook."""

    def __init__(
        self,
        config,
        *,
        channel_size: int = 64,
        num_kernels: int = 2,
        hidden_size: int = 256,
        num_steps: int = 32,
        kernel_size: int = 3,
        cell_dropout_rate: float = 0.1,
        input_embedding_dim: int = 32,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd to preserve the sequence length.")

        self.requires_targets = False
        self.channel_size = int(channel_size)
        self.num_kernels = int(num_kernels)
        self.hidden_size = int(hidden_size)
        self.num_steps = int(num_steps)
        self.kernel_size = int(kernel_size)
        self.cell_dropout_rate = float(cell_dropout_rate)
        self.perception_size = self.channel_size * self.num_kernels

        vocab_size = int(config.input_vocab_size)
        embedding_dim = int(input_embedding_dim)

        self.token_embed = nn.Embedding(vocab_size, embedding_dim)
        self.state_encoder = nn.Linear(embedding_dim, self.channel_size)

        padding = self.kernel_size // 2
        self.perceive = nn.Conv1d(
            self.channel_size,
            self.perception_size,
            kernel_size=self.kernel_size,
            padding=padding,
            groups=self.channel_size,
            bias=False,
        )

        update_input_dim = self.channel_size + self.perception_size
        if use_layer_norm:
            self.update_norm: nn.Module = nn.LayerNorm(update_input_dim)
            self.state_norm: nn.Module = nn.LayerNorm(self.channel_size)
        else:
            self.update_norm = nn.Identity()
            self.state_norm = nn.Identity()

        self.update_mlp = nn.Sequential(
            nn.Linear(update_input_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.channel_size),
        )
        self.output_head = nn.Linear(self.channel_size, int(config.output_vocab_size))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.state_encoder.weight)
        nn.init.zeros_(self.state_encoder.bias)
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

        nn.init.kaiming_uniform_(self.update_mlp[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.update_mlp[0].bias)
        nn.init.xavier_uniform_(self.update_mlp[2].weight)
        nn.init.zeros_(self.update_mlp[2].bias)

        with torch.no_grad():
            weight = torch.zeros_like(self.perceive.weight)
            center = self.kernel_size // 2
            identity_kernel = torch.zeros(self.kernel_size, dtype=weight.dtype)
            identity_kernel[center] = 1.0
            gradient_kernel = torch.zeros(self.kernel_size, dtype=weight.dtype)
            if self.kernel_size >= 3:
                left = max(center - 1, 0)
                right = min(center + 1, self.kernel_size - 1)
                if left != center:
                    gradient_kernel[left] = -0.5
                if right != center:
                    gradient_kernel[right] = 0.5
            kernels = []
            if self.num_kernels >= 1:
                kernels.append(identity_kernel)
            if self.num_kernels >= 2:
                kernels.append(gradient_kernel)
            while len(kernels) < self.num_kernels:
                noise = torch.randn(self.kernel_size, dtype=weight.dtype)
                noise = noise / (noise.norm(p=2) + 1e-6)
                kernels.append(noise)

            for kernel_index in range(self.num_kernels):
                template = kernels[kernel_index].to(weight.device)
                for channel in range(self.channel_size):
                    out_idx = kernel_index * self.channel_size + channel
                    weight[out_idx, 0, : self.kernel_size] = template
            self.perceive.weight.copy_(weight)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del targets  # unused

        if inputs.dtype != torch.long:
            inputs = inputs.long()

        state = self.token_embed(inputs)
        state = self.state_encoder(state).transpose(1, 2)  # (B, C, L)

        for _ in range(self.num_steps):
            perception = self.perceive(state)
            features = torch.cat([state, perception], dim=1).transpose(1, 2)  # (B, L, F)
            features = self.update_norm(features)
            delta = self.update_mlp(features).transpose(1, 2)

            if self.cell_dropout_rate > 0.0 and self.training:
                mask = (torch.rand_like(state[:, :1]) > self.cell_dropout_rate).to(state.dtype)
                delta = delta * mask

            state = state + delta

        features = self.state_norm(state.transpose(1, 2))
        logits = self.output_head(features)
        return logits


__all__ = ["NCA1DSeq2Seq"]
