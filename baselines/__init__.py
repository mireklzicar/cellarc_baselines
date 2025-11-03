from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .neural.cnn.cnn1d import CNN1DSeq2Seq
from .neural.recursive_reasoning import (
    HierarchicalReasoningModel_ACTV1,
    Model_ACTV2,
    TinyRecursiveReasoningModel_ACTV1,
)
from .neural.nca.nca1d import NCA1DSeq2Seq
from .neural.rnn.rnn import RNNModel
from .neural.rnn.stack_rnn import StackRNNCore
from .neural.rnn.tape_rnn import TapeInputLengthJumpCore
from .neural.transformer import make_transformer


@dataclass
class BaselineConfig:
    """Common configuration passed to every baseline factory."""

    input_vocab_size: int
    output_vocab_size: int
    max_seq_len: int
    batch_size: int
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32
    model_kwargs: Mapping[str, Any] = field(default_factory=dict)


class RNNSeq2Seq(nn.Module):
    """Wrapper around the LSTM-based baseline to expose a uniform interface."""

    def __init__(
        self,
        config: BaselineConfig,
        *,
        hidden_size: int = 128,
        input_window: int = 1,
        return_all_outputs: bool = True,
        **rnn_kwargs: Any,
    ) -> None:
        super().__init__()
        self.input_vocab_size = config.input_vocab_size
        self.output_vocab_size = config.output_vocab_size
        self.model = RNNModel(
            output_size=config.output_vocab_size,
            rnn_core="lstm",
            return_all_outputs=return_all_outputs,
            input_window=input_window,
            input_size=config.input_vocab_size,
            hidden_size=hidden_size,
            **rnn_kwargs,
        )
        self.requires_targets = False

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del targets  # unused
        one_hot = F.one_hot(inputs, num_classes=self.input_vocab_size).to(torch.float32)
        logits = self.model(one_hot, input_length=inputs.size(1))
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)
        return logits


class TransformerSeq2Seq(nn.Module):
    """Transformer seq2seq baseline with teacher forcing."""

    def __init__(
        self,
        config: BaselineConfig,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        self.input_vocab_size = config.input_vocab_size
        self.output_vocab_size = config.output_vocab_size
        embedding_dim = int(model_kwargs.pop("embedding_dim", 128))
        num_layers = int(model_kwargs.pop("num_layers", 2))
        num_heads = int(model_kwargs.pop("num_heads", 4))
        self.model = make_transformer(
            output_size=config.output_vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            return_all_outputs=True,
            input_size=config.input_vocab_size,
            **model_kwargs,
        )
        self.requires_targets = True

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if targets is None:
            raise ValueError("TransformerSeq2Seq requires target tokens for teacher forcing.")
        enc = F.one_hot(inputs, num_classes=self.input_vocab_size).to(torch.float32)
        tgt = torch.clamp(targets, min=0)
        dec = F.one_hot(tgt, num_classes=self.output_vocab_size).to(torch.float32)
        return self.model(enc, dec)


class TinyRecursiveSeq2Seq(nn.Module):
    """Tiny recursive reasoning baseline (single ACT step)."""

    def __init__(
        self,
        config: BaselineConfig,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        vocab_size = max(config.input_vocab_size, config.output_vocab_size)
        hidden_size = int(model_kwargs.pop("hidden_size", 128))
        H_layers = int(model_kwargs.pop("H_layers", 1))
        L_layers = int(model_kwargs.pop("L_layers", 1))
        H_cycles = int(model_kwargs.pop("H_cycles", 1))
        L_cycles = int(model_kwargs.pop("L_cycles", 1))
        halt_max_steps = int(model_kwargs.pop("halt_max_steps", 1))
        config_dict = {
            "batch_size": config.batch_size,
            "seq_len": config.max_seq_len,
            "puzzle_emb_ndim": 0,
            "num_puzzle_identifiers": max(1, config.batch_size),
            "puzzle_emb_len": 0,
            "vocab_size": vocab_size,
            "H_cycles": H_cycles,
            "L_cycles": L_cycles,
            "H_layers": H_layers,
            "L_layers": L_layers,
            "hidden_size": hidden_size,
            "expansion": 2.0,
            "num_heads": 4,
            "pos_encodings": "rope",
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "halt_max_steps": halt_max_steps,
            "halt_exploration_prob": 0.0,
            "forward_dtype": "float32",
            "mlp_t": False,
        }
        custom_config = dict(config_dict)
        custom_config.update({k: v for k, v in model_kwargs.items() if v is not None})
        self.model = TinyRecursiveReasoningModel_ACTV1(custom_config)
        self.requires_targets = False

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del targets  # unused
        device = self.model.inner.H_init.device
        inputs = inputs.to(device=device)
        batch = {
            "inputs": inputs.to(torch.int32),
            "puzzle_identifiers": torch.zeros(
                inputs.size(0), dtype=torch.int32, device=device
            ),
        }
        carry = self.model.initial_carry(batch)
        _, outputs = self.model(carry, batch)
        return outputs["logits"]


class HierarchicalReasoningSeq2Seq(nn.Module):
    """Hierarchical reasoning baseline with adaptive computation (HRM)."""

    def __init__(
        self,
        config: BaselineConfig,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        vocab_size = max(config.input_vocab_size, config.output_vocab_size)
        hidden_size = int(model_kwargs.pop("hidden_size", 128))
        H_layers = int(model_kwargs.pop("H_layers", 1))
        L_layers = int(model_kwargs.pop("L_layers", 1))
        H_cycles = int(model_kwargs.pop("H_cycles", 1))
        L_cycles = int(model_kwargs.pop("L_cycles", 1))
        halt_max_steps = int(model_kwargs.pop("halt_max_steps", 1))
        halt_exploration_prob = float(model_kwargs.pop("halt_exploration_prob", 0.0))
        expansion = float(model_kwargs.pop("expansion", 2.0))
        num_heads = int(model_kwargs.pop("num_heads", 4))
        pos_encodings = str(model_kwargs.pop("pos_encodings", "rope"))
        rms_norm_eps = float(model_kwargs.pop("rms_norm_eps", 1e-5))
        rope_theta = float(model_kwargs.pop("rope_theta", 10_000.0))
        forward_dtype = str(model_kwargs.pop("forward_dtype", "float32"))
        mlp_t = bool(model_kwargs.pop("mlp_t", False))
        puzzle_emb_ndim = int(model_kwargs.pop("puzzle_emb_ndim", 0))
        num_puzzle_identifiers = int(
            model_kwargs.pop("num_puzzle_identifiers", max(1, config.batch_size))
        )

        config_dict = {
            "batch_size": config.batch_size,
            "seq_len": config.max_seq_len,
            "puzzle_emb_ndim": puzzle_emb_ndim,
            "num_puzzle_identifiers": num_puzzle_identifiers,
            "vocab_size": vocab_size,
            "H_cycles": H_cycles,
            "L_cycles": L_cycles,
            "H_layers": H_layers,
            "L_layers": L_layers,
            "hidden_size": hidden_size,
            "expansion": expansion,
            "num_heads": num_heads,
            "pos_encodings": pos_encodings,
            "rms_norm_eps": rms_norm_eps,
            "rope_theta": rope_theta,
            "halt_max_steps": halt_max_steps,
            "halt_exploration_prob": halt_exploration_prob,
            "forward_dtype": forward_dtype,
            "mlp_t": mlp_t,
        }
        config_dict.update({k: v for k, v in model_kwargs.items() if v is not None})
        self.model = HierarchicalReasoningModel_ACTV1(config_dict)
        self.requires_targets = False

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del targets  # unused
        device = self.model.inner.H_init.device
        inputs = inputs.to(device=device)
        batch = {
            "inputs": inputs.to(torch.int32),
            "puzzle_identifiers": torch.zeros(
                inputs.size(0), dtype=torch.int32, device=device
            ),
        }
        carry = self.model.initial_carry(batch)
        _, outputs = self.model(carry, batch)
        return outputs["logits"]


class TransformerACTSeq2Seq(nn.Module):
    """Transformer-only variant of the ACT-based recursive reasoning baseline."""

    def __init__(
        self,
        config: BaselineConfig,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        vocab_size = max(config.input_vocab_size, config.output_vocab_size)
        hidden_size = int(model_kwargs.pop("hidden_size", 256))
        H_cycles = int(model_kwargs.pop("H_cycles", 1))
        H_layers = int(model_kwargs.pop("H_layers", 1))
        expansion = float(model_kwargs.pop("expansion", 2.0))
        num_heads = int(model_kwargs.pop("num_heads", 4))
        pos_encodings = str(model_kwargs.pop("pos_encodings", "rope"))
        halt_max_steps = int(model_kwargs.pop("halt_max_steps", 1))
        halt_exploration_prob = float(model_kwargs.pop("halt_exploration_prob", 0.0))
        forward_dtype = str(model_kwargs.pop("forward_dtype", "bfloat16"))
        puzzle_emb_ndim = int(model_kwargs.pop("puzzle_emb_ndim", 0))
        num_puzzle_identifiers = int(
            model_kwargs.pop("num_puzzle_identifiers", max(1, config.batch_size))
        )

        config_dict = {
            "batch_size": config.batch_size,
            "seq_len": config.max_seq_len,
            "puzzle_emb_ndim": puzzle_emb_ndim,
            "num_puzzle_identifiers": num_puzzle_identifiers,
            "vocab_size": vocab_size,
            "H_cycles": H_cycles,
            "H_layers": H_layers,
            "hidden_size": hidden_size,
            "expansion": expansion,
            "num_heads": num_heads,
            "pos_encodings": pos_encodings,
            "halt_max_steps": halt_max_steps,
            "halt_exploration_prob": halt_exploration_prob,
            "forward_dtype": forward_dtype,
        }

        optional_fields = (
            "rms_norm_eps",
            "rope_theta",
            "act_enabled",
            "act_inference",
        )
        for field in optional_fields:
            value = model_kwargs.pop(field, None)
            if value is not None:
                config_dict[field] = value

        config_dict.update({k: v for k, v in model_kwargs.items() if v is not None})
        self.model = Model_ACTV2(config_dict)
        self.requires_targets = False

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del targets  # unused
        device = self.model.inner.H_init.device
        inputs = inputs.to(device=device)
        batch = {
            "inputs": inputs.to(torch.int32),
            "puzzle_identifiers": torch.zeros(
                inputs.size(0), dtype=torch.int32, device=device
            ),
        }
        carry = self.model.initial_carry(batch)
        _, outputs = self.model(carry, batch)
        return outputs["logits"]


class TapeRNNSeq2Seq(nn.Module):
    """Sequence-to-sequence wrapper for the differentiable tape RNN core."""

    def __init__(
        self,
        config: BaselineConfig,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        self.input_vocab_size = config.input_vocab_size
        self.output_vocab_size = config.output_vocab_size

        core_kwargs = dict(model_kwargs)
        memory_cell_size = int(core_kwargs.pop("memory_cell_size", 128))
        memory_size = int(core_kwargs.pop("memory_size", 32))
        n_tapes = int(core_kwargs.pop("n_tapes", 1))
        mlp_layers_size = core_kwargs.pop("mlp_layers_size", (64, 64))
        if mlp_layers_size is None:
            mlp_layers = ()
        elif isinstance(mlp_layers_size, Sequence) and not isinstance(mlp_layers_size, (str, bytes)):
            mlp_layers = tuple(int(layer) for layer in mlp_layers_size)
        else:
            mlp_layers = (int(mlp_layers_size),)
        inner_core = str(core_kwargs.pop("inner_core", "lstm"))
        hidden_size = int(core_kwargs.pop("hidden_size", 256))
        input_size = int(core_kwargs.pop("input_size", config.input_vocab_size))
        input_window = int(core_kwargs.pop("input_window", 1))

        tape_core = TapeInputLengthJumpCore(
            memory_cell_size=memory_cell_size,
            memory_size=memory_size,
            n_tapes=n_tapes,
            mlp_layers_size=mlp_layers,
            inner_core=inner_core,
            hidden_size=hidden_size,
            input_size=input_size,
            **core_kwargs,
        )
        self.model = RNNModel(
            output_size=config.output_vocab_size,
            rnn_core=tape_core,
            return_all_outputs=True,
            input_window=input_window,
            input_size=input_size,
            hidden_size=hidden_size,
        )
        self.requires_targets = False

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del targets  # unused
        one_hot = F.one_hot(inputs, num_classes=self.input_vocab_size).to(torch.float32)
        logits = self.model(one_hot, input_length=inputs.size(1))
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)
        return logits


class StackRNNSeq2Seq(nn.Module):
    """Sequence-to-sequence wrapper for the differentiable stack RNN core."""

    def __init__(
        self,
        config: BaselineConfig,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        self.input_vocab_size = config.input_vocab_size
        self.output_vocab_size = config.output_vocab_size

        core_kwargs = dict(model_kwargs)
        stack_cell_size = int(core_kwargs.pop("stack_cell_size", 128))
        stack_size = int(core_kwargs.pop("stack_size", 32))
        n_stacks = int(core_kwargs.pop("n_stacks", 1))
        inner_core = str(core_kwargs.pop("inner_core", "lstm"))
        hidden_size = int(core_kwargs.pop("hidden_size", 256))
        input_size = int(core_kwargs.pop("input_size", config.input_vocab_size))
        input_window = int(core_kwargs.pop("input_window", 1))

        stack_core = StackRNNCore(
            stack_cell_size=stack_cell_size,
            stack_size=stack_size,
            n_stacks=n_stacks,
            inner_core=inner_core,
            hidden_size=hidden_size,
            input_size=input_size,
            **core_kwargs,
        )
        self.model = RNNModel(
            output_size=config.output_vocab_size,
            rnn_core=stack_core,
            return_all_outputs=True,
            input_window=input_window,
            input_size=input_size,
            hidden_size=hidden_size,
        )
        self.requires_targets = False

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del targets  # unused
        one_hot = F.one_hot(inputs, num_classes=self.input_vocab_size).to(torch.float32)
        logits = self.model(one_hot, input_length=inputs.size(1))
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)
        return logits


def _build_rnn(config: BaselineConfig) -> nn.Module:
    return RNNSeq2Seq(config, **dict(config.model_kwargs)).to(config.device, dtype=config.dtype)


def _build_transformer(config: BaselineConfig) -> nn.Module:
    return TransformerSeq2Seq(config, **dict(config.model_kwargs)).to(config.device, dtype=config.dtype)


def _build_cnn(config: BaselineConfig) -> nn.Module:
    return CNN1DSeq2Seq(
        config.input_vocab_size,
        config.output_vocab_size,
        **dict(config.model_kwargs),
    ).to(config.device, dtype=config.dtype)


def _build_tiny_recursive(config: BaselineConfig) -> nn.Module:
    model = TinyRecursiveSeq2Seq(config, **dict(config.model_kwargs))
    return model.to(config.device, dtype=config.dtype)


def _build_hierarchical_recursive(config: BaselineConfig) -> nn.Module:
    model = HierarchicalReasoningSeq2Seq(config, **dict(config.model_kwargs))
    return model.to(config.device, dtype=config.dtype)


def _build_transformer_act(config: BaselineConfig) -> nn.Module:
    model = TransformerACTSeq2Seq(config, **dict(config.model_kwargs))
    return model.to(config.device, dtype=config.dtype)


def _build_tape_rnn(config: BaselineConfig) -> nn.Module:
    model = TapeRNNSeq2Seq(config, **dict(config.model_kwargs))
    return model.to(config.device, dtype=config.dtype)


def _build_stack_rnn(config: BaselineConfig) -> nn.Module:
    model = StackRNNSeq2Seq(config, **dict(config.model_kwargs))
    return model.to(config.device, dtype=config.dtype)


def _build_nca1d(config: BaselineConfig) -> nn.Module:
    model = NCA1DSeq2Seq(config, **dict(config.model_kwargs))
    return model.to(config.device, dtype=config.dtype)


BASELINE_REGISTRY: Dict[str, Callable[[BaselineConfig], nn.Module]] = {
    "rnn": _build_rnn,
    "transformer": _build_transformer,
    "cnn1d": _build_cnn,
    "tiny_recursive": _build_tiny_recursive,
    "trm": _build_tiny_recursive,
    "hrm": _build_hierarchical_recursive,
    "transformer_act": _build_transformer_act,
    "tape_rnn": _build_tape_rnn,
    "stack_rnn": _build_stack_rnn,
    "nca1d": _build_nca1d,
}


def get_baseline_registry() -> Mapping[str, Callable[[BaselineConfig], nn.Module]]:
    """Return a read-only view of available baselines."""
    return dict(BASELINE_REGISTRY)


def create_baseline(name: str, config: BaselineConfig) -> nn.Module:
    try:
        builder = BASELINE_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown baseline {name!r}. Available: {list(BASELINE_REGISTRY)}") from exc
    return builder(config)


__all__ = [
    "BaselineConfig",
    "BASELINE_REGISTRY",
    "CNN1DSeq2Seq",
    "RNNSeq2Seq",
    "TinyRecursiveSeq2Seq",
    "HierarchicalReasoningSeq2Seq",
    "TransformerACTSeq2Seq",
    "TapeRNNSeq2Seq",
    "StackRNNSeq2Seq",
    "TransformerSeq2Seq",
    "NCA1DSeq2Seq",
    "create_baseline",
    "get_baseline_registry",
]
