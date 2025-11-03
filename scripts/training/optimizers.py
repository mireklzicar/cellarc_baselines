"""Custom optimizer utilities used across training pipelines."""

from __future__ import annotations

from typing import Iterable, Optional

import torch


class AdamATan2(torch.optim.AdamW):
    """A light-weight AdamW wrapper used as a stand-in for Adam-Atan2.

    The original recursive reasoning training loops relied on the Adam-Atan2
    optimizer, which is not readily available in this repository. To preserve
    compatibility while keeping the training code simple, we delegate to the
    PyTorch AdamW implementation but expose the same API surface so the call
    sites remain identical.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        *,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
