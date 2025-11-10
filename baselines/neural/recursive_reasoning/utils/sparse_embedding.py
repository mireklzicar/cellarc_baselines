from typing import Optional, Union

import torch
from torch import nn
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, ParamsT

from .common import trunc_normal_init_


class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self._local_grad: Optional[torch.Tensor] = None
        self._active_rows: int = 0
        self._embedding_dim = embedding_dim

        # Real Weights
        # Truncated LeCun normal init
        self.weights = nn.Buffer(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std), persistent=True
        )

        # Local weights and IDs
        # Local embeddings, with gradient, not persistent
        self.local_weights = nn.Buffer(
            torch.zeros(batch_size, embedding_dim, requires_grad=True), persistent=False
        )
        # Local embedding IDs, not persistent
        self.local_ids = nn.Buffer(torch.zeros(batch_size, dtype=torch.int32), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            # Test mode, no gradient
            return self.weights[inputs].to(self.cast_to)
        
        # Ensure tensor dtypes for indexing and capture original layout
        original_shape = inputs.shape
        flat_inputs = inputs.reshape(-1).to(device=self.weights.device, dtype=torch.long)
        current_batch = flat_inputs.shape[0]

        if current_batch == 0:
            self._active_rows = 0
            self._local_grad = None
            return self.local_weights[:0].to(self.cast_to).reshape(*original_shape, self._embedding_dim)

        # Expand local buffers if the current batch exceeds the preallocated capacity
        if current_batch > self.local_weights.shape[0]:
            new_local_weights = torch.zeros(
                (current_batch, self._embedding_dim),
                dtype=self.local_weights.dtype,
                device=self.local_weights.device,
                requires_grad=True,
            )
            new_local_ids = torch.zeros(
                current_batch,
                dtype=self.local_ids.dtype,
                device=self.local_ids.device,
            )
            if self.local_weights.numel() > 0:
                new_local_weights[: self.local_weights.shape[0]].copy_(self.local_weights)
                new_local_ids[: self.local_ids.shape[0]].copy_(self.local_ids)
            self.register_buffer("local_weights", new_local_weights, persistent=False)
            self.register_buffer("local_ids", new_local_ids, persistent=False)

        # Populate the active slice with the embedding weights for the current ids
        with torch.no_grad():
            gathered = self.weights[flat_inputs].view(current_batch, self._embedding_dim)
            self.local_weights[:current_batch].copy_(gathered)
            self.local_ids[:current_batch].copy_(flat_inputs.to(self.local_ids.dtype))

        self._active_rows = current_batch
        self._local_grad = None

        local_output = self.local_weights[:current_batch].to(self.cast_to)
        output = local_output.reshape(*original_shape, self._embedding_dim)

        if not torch.is_grad_enabled():
            return output

        if not output.requires_grad:
            output = output.detach().requires_grad_(True)

        def _store_local_grad(grad: torch.Tensor) -> torch.Tensor:
            self._local_grad = grad.reshape(current_batch, self._embedding_dim).to(self.weights.dtype)
            return grad

        output.register_hook(_store_local_grad)
        return output

    @property
    def local_grad(self) -> Optional[torch.Tensor]:
        return self._local_grad

    def clear_local_grad(self) -> None:
        self._local_grad = None
        self._active_rows = 0

    @property
    def active_rows(self) -> int:
        return self._active_rows


class CastedSparseEmbeddingSignSGD_Distributed(Optimizer):
    def __init__(
        self,
        params: ParamsT,

        world_size: int,
        lr: Union[float, torch.Tensor] = 1e-3,
        weight_decay: float = 1e-2,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            world_size=world_size
        )
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):  # type: ignore
        for group in self.param_groups:
            # Find the sparse embedding weights
            local_weights_grad = None
            local_ids = None
            weights = None
            
            assert len(group["params"]) == 3
            for p in group["params"]:
                if p.requires_grad:
                    local_weights_grad = p.grad
                elif p.ndim == 1:
                    local_ids = p
                elif p.ndim == 2:
                    weights = p
                else:
                    assert False
                
            assert local_ids is not None
            assert weights is not None
        
            # Apply SignSGD
            # Adam ≈ SignSGD if gradient is very sparse
            if local_weights_grad is not None:
                _sparse_emb_signsgd_dist(
                    local_weights_grad,
                    local_ids,
                    weights,
                    
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    world_size=group["world_size"]
                )


def _sparse_emb_signsgd_dist(
    local_weights_grad: torch.Tensor,
    local_ids: torch.Tensor,
    weights: torch.Tensor,
    
    lr: float,
    weight_decay: float,
    world_size: int
) -> None:
    N, D = local_weights_grad.shape
    
    # All-gather
    all_weights_grad = local_weights_grad
    all_ids = local_ids

    if world_size > 1:
        all_weights_grad = torch.empty((world_size * N, D), dtype=local_weights_grad.dtype, device=local_weights_grad.device)
        all_ids = torch.empty(world_size * N,               dtype=local_ids.dtype,          device=local_ids.device)
    
        dist.all_gather_into_tensor(all_weights_grad, local_weights_grad)
        dist.all_gather_into_tensor(all_ids,          local_ids)

    # Unique
    grad_ids, inv = all_ids.unique(return_inverse=True)

    grad = torch.zeros((grad_ids.shape[0], D), dtype=all_weights_grad.dtype, device=all_weights_grad.device)
    grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, D), all_weights_grad)

    # SignSGD with decoupled weight decay
    p = weights[grad_ids]

    p.mul_(1.0 - lr * weight_decay).add_(torch.sign(grad), alpha=-lr)

    # Write updated slices back
    weights[grad_ids] = p
