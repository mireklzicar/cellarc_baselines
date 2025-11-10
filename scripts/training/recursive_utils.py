"""Utilities supporting recursive reasoning embedding training."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import nn

try:
    from baselines.neural.recursive_reasoning.utils.sparse_embedding import (
        CastedSparseEmbedding,
    )
except ImportError as exc:  # pragma: no cover - dependency should exist
    raise RuntimeError(
        "CastedSparseEmbedding is required for recursive embedding training."
    ) from exc


class PuzzleEmbeddingOptimizer:
    """Manual optimizer applying SignSGD-style updates to puzzle embeddings."""

    def __init__(
        self,
        embeddings: Sequence[CastedSparseEmbedding],
        *,
        lr: float,
        weight_decay: float,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        self._embeddings = [module for module in embeddings]
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self._process_group = process_group
        self._world_size = 1
        if process_group is not None:
            try:
                self._world_size = dist.get_world_size(process_group)
            except Exception:
                self._world_size = 1

    def state_dict(self) -> Dict[str, float]:
        return {"lr": self.lr, "weight_decay": self.weight_decay, "world_size": float(self._world_size)}

    def load_state_dict(self, state_dict: Mapping[str, float]) -> None:
        self.lr = float(state_dict.get("lr", self.lr))
        self.weight_decay = float(state_dict.get("weight_decay", self.weight_decay))
        if "world_size" in state_dict:
            self._world_size = int(state_dict["world_size"])

    @torch.no_grad()
    def step(self) -> None:
        if not self._embeddings or self.lr == 0.0:
            return

        for embedding in self._embeddings:
            grad = getattr(embedding, "local_grad", None)
            if grad is None:
                continue
            if grad.numel() == 0:
                continue

            active_rows = getattr(embedding, "active_rows", None)
            if active_rows is None:
                active_rows = grad.shape[0]
            if active_rows == 0:
                continue

            puzzle_ids = embedding.local_ids[:active_rows]
            grad = grad[:active_rows]

            if puzzle_ids.numel() == 0:
                continue

            unique_ids, inverse = torch.unique(puzzle_ids, return_inverse=True)
            grad_sum = torch.zeros(
                (unique_ids.size(0), grad.size(1)),
                dtype=grad.dtype,
                device=grad.device,
            )
            scatter_index = inverse.unsqueeze(-1).expand(-1, grad.size(1))
            grad_sum.scatter_add_(0, scatter_index, grad)

            weights = embedding.weights
            if self._process_group is not None and self._world_size > 1:
                unique_ids, grad_sum = self._all_gather_sparse_grads(
                    unique_ids=unique_ids,
                    grad_sum=grad_sum,
                    device=weights.device,
                )
                if unique_ids.numel() == 0:
                    continue

            unique_ids_long = unique_ids.to(dtype=torch.long)
            updated = weights.index_select(0, unique_ids_long)
            if self.weight_decay:
                updated = updated.mul(1.0 - self.lr * self.weight_decay)
            updated.add_(torch.sign(grad_sum), alpha=-self.lr)
            weights.index_copy_(0, unique_ids_long, updated)

            if hasattr(embedding, "clear_local_grad"):
                embedding.clear_local_grad()

    def zero_grad(self) -> None:
        for embedding in self._embeddings:
            if hasattr(embedding, "clear_local_grad"):
                embedding.clear_local_grad()

    def _all_gather_sparse_grads(
        self,
        *,
        unique_ids: torch.Tensor,
        grad_sum: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gathered: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * self._world_size
        payload = (
            unique_ids.detach().to(device="cpu"),
            grad_sum.detach().to(device="cpu"),
        )
        dist.all_gather_object(gathered, payload, group=self._process_group)

        merged_ids: List[torch.Tensor] = []
        merged_grads: List[torch.Tensor] = []
        for item in gathered:
            if item is None:
                continue
            ids_tensor, grad_tensor = item
            if ids_tensor.numel() == 0:
                continue
            merged_ids.append(ids_tensor.to(device=device))
            merged_grads.append(grad_tensor.to(device=device))

        if not merged_ids:
            return (
                unique_ids.new_zeros((0,), device=device),
                grad_sum.new_zeros((0, grad_sum.size(1)), device=device),
            )

        all_ids = torch.cat(merged_ids, dim=0)
        all_grads = torch.cat(merged_grads, dim=0)
        merged_unique, inverse = torch.unique(all_ids, return_inverse=True)
        aggregated = torch.zeros(
            (merged_unique.size(0), all_grads.size(1)),
            dtype=all_grads.dtype,
            device=device,
        )
        aggregated.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, all_grads.size(1)), all_grads)
        return merged_unique, aggregated

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return bool(self._embeddings)


class EMAHelper:
    """Exponential moving average tracker for model parameters."""

    def __init__(self, decay: float) -> None:
        if not 0.0 < decay <= 1.0:
            raise ValueError("EMA decay must lie in (0, 1].")
        self.decay = float(decay)
        self._shadow: Dict[str, torch.Tensor] = {}

    def state_dict(self) -> Dict[str, object]:
        return {
            "decay": self.decay,
            "shadow": {name: tensor.clone() for name, tensor in self._shadow.items()},
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        decay = state_dict.get("decay", self.decay)
        self.decay = float(decay)
        shadow = state_dict.get("shadow")
        if isinstance(shadow, Mapping):
            self._shadow = {
                str(name): tensor.clone()
                for name, tensor in shadow.items()
                if isinstance(tensor, torch.Tensor)
            }

    @torch.no_grad()
    def register(self, module: nn.Module) -> None:
        self._shadow.clear()
        for name, param in module.named_parameters():
            if param.requires_grad:
                self._shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            value = self._shadow.get(name)
            if value is None:
                self._shadow[name] = param.detach().clone()
                continue
            value.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_to_shadow(self, module: nn.Module) -> Dict[str, torch.Tensor]:
        backup: Dict[str, torch.Tensor] = {}
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            shadow_value = self._shadow.get(name)
            if shadow_value is None:
                continue
            backup[name] = param.data.clone()
            param.data.copy_(shadow_value)
        return backup

    @torch.no_grad()
    def restore(self, module: nn.Module, backup: Mapping[str, torch.Tensor]) -> None:
        for name, param in module.named_parameters():
            saved = backup.get(name)
            if saved is not None:
                param.data.copy_(saved)
