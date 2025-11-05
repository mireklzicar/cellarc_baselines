from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, List

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

from .neural.transformer import make_transformer, make_transformer_encoder


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
    pad_token_id: Optional[int] = None


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

        puzzle_emb_ndim = int(rnn_kwargs.pop("puzzle_emb_ndim", 0))
        num_puzzle_identifiers = int(rnn_kwargs.pop("num_puzzle_identifiers", 0))
        puzzle_emb_init_std = float(rnn_kwargs.pop("puzzle_emb_init_std", 0.02))
        puzzle_proj_init_std = float(rnn_kwargs.pop("puzzle_proj_init_std", 0.02))

        self.puzzle_embedding: Optional[nn.Embedding] = None
        self.puzzle_projector: Optional[nn.Linear] = None
        if puzzle_emb_ndim > 0 and num_puzzle_identifiers > 0:
            self.puzzle_embedding = nn.Embedding(
                num_puzzle_identifiers, puzzle_emb_ndim
            )
            nn.init.trunc_normal_(self.puzzle_embedding.weight, std=puzzle_emb_init_std)
            self.puzzle_projector = nn.Linear(
                puzzle_emb_ndim,
                self.input_vocab_size,
                bias=True,
            )
            nn.init.trunc_normal_(self.puzzle_projector.weight, std=puzzle_proj_init_std)
            if self.puzzle_projector.bias is not None:
                nn.init.zeros_(self.puzzle_projector.bias)

        self.model = RNNModel(
            output_size=config.output_vocab_size,
            rnn_core="lstm",
            return_all_outputs=return_all_outputs,
            input_window=input_window,
            input_size=config.input_vocab_size,
            hidden_size=hidden_size,
            **rnn_kwargs,
        )
        self.requires_puzzle_identifiers = self.puzzle_embedding is not None
        self.requires_targets = False

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        puzzle_identifiers: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del targets  # unused
        del loss_mask  # unused
        one_hot = F.one_hot(inputs, num_classes=self.input_vocab_size).to(torch.float32)
        if self.puzzle_embedding is not None and self.puzzle_projector is not None:
            if puzzle_identifiers is None:
                raise ValueError(
                    "RNNSeq2Seq expects puzzle identifiers when puzzle embeddings are enabled."
                )
            puzzle_identifiers = puzzle_identifiers.to(inputs.device)
            puzzle_vectors = self.puzzle_embedding(puzzle_identifiers)
            puzzle_bias = self.puzzle_projector(puzzle_vectors).to(one_hot.dtype)
            one_hot = one_hot + puzzle_bias.unsqueeze(1)
        logits = self.model(one_hot, input_length=inputs.size(1))
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)
        return logits


class RNNSeq2SeqAutoregressive(nn.Module):
    """Seq2seq LSTM baseline with teacher forcing during training and autoregressive decoding."""

    def __init__(
        self,
        config: BaselineConfig,
        **model_kwargs: Any,
    ) -> None:
        super().__init__()
        self.input_vocab_size = config.input_vocab_size
        self.output_vocab_size = config.output_vocab_size
        hidden_size = int(model_kwargs.pop("hidden_size", 128))
        num_layers = max(1, int(model_kwargs.pop("num_layers", 1)))
        self.teacher_forcing = bool(model_kwargs.pop("teacher_forcing", True))

        puzzle_emb_ndim = int(model_kwargs.pop("puzzle_emb_ndim", 0))
        num_puzzle_identifiers = int(model_kwargs.pop("num_puzzle_identifiers", 0))
        puzzle_emb_init_std = float(model_kwargs.pop("puzzle_emb_init_std", 0.02))
        puzzle_proj_init_std = float(model_kwargs.pop("puzzle_proj_init_std", 0.02))

        self.puzzle_embedding: Optional[nn.Embedding] = None
        self.puzzle_projector: Optional[nn.Linear] = None
        if puzzle_emb_ndim > 0 and num_puzzle_identifiers > 0:
            self.puzzle_embedding = nn.Embedding(
                num_puzzle_identifiers, puzzle_emb_ndim
            )
            nn.init.trunc_normal_(self.puzzle_embedding.weight, std=puzzle_emb_init_std)
            self.puzzle_projector = nn.Linear(
                puzzle_emb_ndim,
                hidden_size,
                bias=True,
            )
            nn.init.trunc_normal_(self.puzzle_projector.weight, std=puzzle_proj_init_std)
            if self.puzzle_projector.bias is not None:
                nn.init.zeros_(self.puzzle_projector.bias)

        self.input_embedding = nn.Embedding(self.input_vocab_size, hidden_size)
        self.output_embedding = nn.Embedding(self.output_vocab_size + 1, hidden_size)
        self.encoder = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, self.output_vocab_size)

        self.start_token_id = self.output_vocab_size
        self.requires_puzzle_identifiers = self.puzzle_embedding is not None
        self.requires_targets = bool(self.teacher_forcing)
        self.accepts_loss_mask = True

    def train(self, mode: bool = True) -> "RNNSeq2SeqAutoregressive":
        super().train(mode)
        if self.teacher_forcing:
            self.requires_targets = mode
        else:
            self.requires_targets = False
        return self

    def _apply_puzzle_bias(
        self,
        embeddings: torch.Tensor,
        puzzle_identifiers: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.puzzle_embedding is None or self.puzzle_projector is None:
            return embeddings
        if puzzle_identifiers is None:
            raise ValueError(
                "RNNSeq2SeqAutoregressive expects puzzle identifiers when puzzle embeddings are enabled."
            )
        puzzle_vectors = self.puzzle_embedding(puzzle_identifiers)
        puzzle_bias = self.puzzle_projector(puzzle_vectors).to(embeddings.dtype)
        return embeddings + puzzle_bias.unsqueeze(1)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        puzzle_identifiers: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = inputs.device
        if puzzle_identifiers is not None:
            puzzle_identifiers = puzzle_identifiers.to(device)

        encoder_inputs = self.input_embedding(inputs.to(device))
        encoder_inputs = self._apply_puzzle_bias(encoder_inputs, puzzle_identifiers)
        _, encoder_state = self.encoder(encoder_inputs)

        if self.teacher_forcing and self.training:
            if targets is None:
                raise ValueError(
                    "RNNSeq2SeqAutoregressive requires targets for teacher forcing during training."
                )
            target_tokens = torch.clamp(targets.to(device), min=0)
            start_token = torch.full(
                (target_tokens.size(0), 1),
                self.start_token_id,
                dtype=torch.long,
                device=device,
            )
            decoder_input_tokens = torch.cat(
                (start_token, target_tokens[:, :-1]),
                dim=1,
            )
            decoder_inputs = self.output_embedding(decoder_input_tokens)
            decoder_outputs, _ = self.decoder(decoder_inputs, encoder_state)
            return self.output_layer(decoder_outputs)

        # Autoregressive decoding (evaluation or teacher forcing disabled).
        if targets is not None:
            template = torch.clamp(targets.to(device), min=0).long()
            sequence_length = template.shape[1]
        else:
            template = None
            sequence_length = inputs.shape[1]

        if loss_mask is None:
            predict_mask = torch.ones(
                (inputs.size(0), sequence_length),
                dtype=torch.bool,
                device=device,
            )
        else:
            predict_mask = loss_mask.to(device=device, dtype=torch.bool)

        hidden_state = encoder_state
        next_token = torch.full(
            (inputs.size(0), 1),
            self.start_token_id,
            dtype=torch.long,
            device=device,
        )
        logits_per_step: List[torch.Tensor] = []

        for t in range(sequence_length):
            decoder_input = self.output_embedding(next_token)
            decoder_output, hidden_state = self.decoder(decoder_input, hidden_state)
            step_logits = self.output_layer(decoder_output[:, -1, :])
            logits_per_step.append(step_logits.unsqueeze(1))

            step_pred = step_logits.argmax(dim=-1)
            if template is not None:
                template_tokens = template[:, t]
                step_pred = torch.where(
                    predict_mask[:, t],
                    step_pred,
                    template_tokens,
                )
            next_token = step_pred.unsqueeze(1)

        return torch.cat(logits_per_step, dim=1)
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
        teacher_forcing = bool(model_kwargs.pop("teacher_forcing", True))
        puzzle_emb_ndim = int(model_kwargs.pop("puzzle_emb_ndim", 0))
        num_puzzle_identifiers = int(model_kwargs.pop("num_puzzle_identifiers", 0))
        puzzle_emb_init_std = float(model_kwargs.pop("puzzle_emb_init_std", 0.02))
        puzzle_proj_init_std = float(model_kwargs.pop("puzzle_proj_init_std", 0.02))

        self.teacher_forcing = teacher_forcing
        self.puzzle_embedding: Optional[nn.Embedding] = None
        self.puzzle_projector: Optional[nn.Linear] = None
        if puzzle_emb_ndim > 0 and num_puzzle_identifiers > 0:
            self.puzzle_embedding = nn.Embedding(
                num_puzzle_identifiers, puzzle_emb_ndim
            )
            nn.init.trunc_normal_(self.puzzle_embedding.weight, std=puzzle_emb_init_std)
            self.puzzle_projector = nn.Linear(
                puzzle_emb_ndim,
                self.input_vocab_size,
                bias=True,
            )
            nn.init.trunc_normal_(self.puzzle_projector.weight, std=puzzle_proj_init_std)
            if self.puzzle_projector.bias is not None:
                nn.init.zeros_(self.puzzle_projector.bias)

        self.model = make_transformer(
            output_size=config.output_vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            return_all_outputs=True,
            input_size=config.input_vocab_size,
            **model_kwargs,
        )
        self.requires_puzzle_identifiers = self.puzzle_embedding is not None
        self.requires_targets = bool(teacher_forcing)
        self.accepts_loss_mask = True

    def train(self, mode: bool = True) -> "TransformerSeq2Seq":
        super().train(mode)
        if self.teacher_forcing:
            self.requires_targets = mode
        else:
            self.requires_targets = False
        return self

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        puzzle_identifiers: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = inputs.device
        if self.puzzle_embedding is not None:
            if puzzle_identifiers is None:
                raise ValueError(
                    "TransformerSeq2Seq expects puzzle identifiers when puzzle embeddings are enabled."
                )
            puzzle_identifiers = puzzle_identifiers.to(device)

        encoder_inputs = F.one_hot(inputs, num_classes=self.input_vocab_size).to(torch.float32)
        if self.puzzle_embedding is not None and self.puzzle_projector is not None:
            puzzle_vectors = self.puzzle_embedding(puzzle_identifiers)
            puzzle_token = self.puzzle_projector(puzzle_vectors).to(encoder_inputs.dtype)
            encoder_inputs = torch.cat((puzzle_token.unsqueeze(1), encoder_inputs), dim=1)

        if self.teacher_forcing and self.training:
            if targets is None:
                raise ValueError("TransformerSeq2Seq requires target tokens for teacher forcing during training.")
            tgt = torch.clamp(targets, min=0)
            decoder_inputs = F.one_hot(tgt, num_classes=self.output_vocab_size).to(torch.float32)
            return self.model(encoder_inputs, decoder_inputs)

        return self._autoregressive_decode(
            encoder_inputs=encoder_inputs,
            targets=targets,
            loss_mask=loss_mask,
        )

    def _autoregressive_decode(
        self,
        *,
        encoder_inputs: torch.Tensor,
        targets: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Greedy decoding without using ground-truth targets."""
        device = encoder_inputs.device
        batch_size, _, _ = encoder_inputs.shape
        encoded = self.model.transformer._encoder(encoder_inputs)

        if targets is not None:
            template = torch.clamp(targets.to(device), min=0).long()
        else:
            sequence_length = encoder_inputs.shape[1]
            template = torch.zeros(
                (batch_size, sequence_length),
                dtype=torch.long,
                device=device,
            )

        if loss_mask is None:
            predict_mask = torch.ones_like(template, dtype=torch.bool)
        else:
            predict_mask = loss_mask.to(device=device, dtype=torch.bool)

        generated = template.clone()
        generated[predict_mask] = 0

        sequence_length = generated.shape[1]
        for t in range(sequence_length):
            decoder_input = F.one_hot(
                torch.clamp(generated, min=0),
                num_classes=self.output_vocab_size,
            ).to(encoder_inputs.dtype)

            decoder_output = self.model.transformer._decoder(encoded, decoder_input)
            logits = self.model.linear(decoder_output)
            update_mask = predict_mask[:, t]
            if update_mask.any():
                step_preds = logits[:, t, :].argmax(dim=-1)
                generated[update_mask, t] = step_preds[update_mask]
            elif targets is not None:
                generated[:, t] = template[:, t]

        final_decoder_input = F.one_hot(
            torch.clamp(generated, min=0),
            num_classes=self.output_vocab_size,
        ).to(encoder_inputs.dtype)
        final_decoder_output = self.model.transformer._decoder(encoded, final_decoder_input)
        return self.model.linear(final_decoder_output)


class TransformerEncoderSeq2Seq(nn.Module):
    """Transformer encoder-only baseline without teacher forcing."""

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
        puzzle_emb_ndim = int(model_kwargs.pop("puzzle_emb_ndim", 0))
        num_puzzle_identifiers = int(model_kwargs.pop("num_puzzle_identifiers", 0))
        puzzle_emb_init_std = float(model_kwargs.pop("puzzle_emb_init_std", 0.02))
        puzzle_proj_init_std = float(model_kwargs.pop("puzzle_proj_init_std", 0.02))

        self.puzzle_embedding: Optional[nn.Embedding] = None
        self.puzzle_projector: Optional[nn.Linear] = None
        if puzzle_emb_ndim > 0 and num_puzzle_identifiers > 0:
            self.puzzle_embedding = nn.Embedding(
                num_puzzle_identifiers, puzzle_emb_ndim
            )
            nn.init.trunc_normal_(self.puzzle_embedding.weight, std=puzzle_emb_init_std)
            self.puzzle_projector = nn.Linear(
                puzzle_emb_ndim,
                self.input_vocab_size,
                bias=True,
            )
            nn.init.trunc_normal_(self.puzzle_projector.weight, std=puzzle_proj_init_std)
            if self.puzzle_projector.bias is not None:
                nn.init.zeros_(self.puzzle_projector.bias)

        self.model = make_transformer_encoder(
            output_size=config.output_vocab_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            return_all_outputs=True,
            input_size=config.input_vocab_size,
            **model_kwargs,
        )

        self.requires_puzzle_identifiers = self.puzzle_embedding is not None
        self.requires_targets = False
        self.accepts_loss_mask = False

    def forward(
        self,
        inputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        puzzle_identifiers: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del targets, loss_mask  # not used
        encoder_inputs = F.one_hot(inputs, num_classes=self.input_vocab_size).to(torch.float32)
        if self.puzzle_embedding is not None and self.puzzle_projector is not None:
            if puzzle_identifiers is None:
                raise ValueError(
                    "TransformerEncoderSeq2Seq expects puzzle identifiers when puzzle embeddings are enabled."
                )
            puzzle_identifiers = puzzle_identifiers.to(inputs.device)
            puzzle_vectors = self.puzzle_embedding(puzzle_identifiers)
            puzzle_bias = self.puzzle_projector(puzzle_vectors).to(encoder_inputs.dtype)
            encoder_inputs = encoder_inputs + puzzle_bias.unsqueeze(1)
        return self.model(encoder_inputs)


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
        *,
        puzzle_identifiers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del targets  # unused
        device = self.model.inner.H_init.device
        inputs = inputs.to(device=device)
        if puzzle_identifiers is None:
            puzzle_identifiers = torch.zeros(
                inputs.size(0), dtype=torch.int32, device=device
            )
        else:
            puzzle_identifiers = puzzle_identifiers.to(
                device=device, dtype=torch.int32
            )
        batch = {
            "inputs": inputs.to(torch.int32),
            "puzzle_identifiers": puzzle_identifiers,
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
        *,
        puzzle_identifiers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del targets  # unused
        device = self.model.inner.H_init.device
        inputs = inputs.to(device=device)
        if puzzle_identifiers is None:
            puzzle_identifiers = torch.zeros(
                inputs.size(0), dtype=torch.int32, device=device
            )
        else:
            puzzle_identifiers = puzzle_identifiers.to(
                device=device, dtype=torch.int32
            )
        batch = {
            "inputs": inputs.to(torch.int32),
            "puzzle_identifiers": puzzle_identifiers,
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
        *,
        puzzle_identifiers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del targets  # unused
        device = self.model.inner.H_init.device
        inputs = inputs.to(device=device)
        if puzzle_identifiers is None:
            puzzle_identifiers = torch.zeros(
                inputs.size(0), dtype=torch.int32, device=device
            )
        else:
            puzzle_identifiers = puzzle_identifiers.to(
                device=device, dtype=torch.int32
            )
        batch = {
            "inputs": inputs.to(torch.int32),
            "puzzle_identifiers": puzzle_identifiers,
        }
        carry = self.model.initial_carry(batch)
        _, outputs = self.model(carry, batch)
        return outputs["logits"]


def _build_rnn(config: BaselineConfig) -> nn.Module:
    return RNNSeq2Seq(config, **dict(config.model_kwargs)).to(config.device, dtype=config.dtype)


def _build_transformer(config: BaselineConfig) -> nn.Module:
    return TransformerEncoderSeq2Seq(config, **dict(config.model_kwargs)).to(config.device, dtype=config.dtype)


def _build_transformer_ar(config: BaselineConfig) -> nn.Module:
    kwargs = dict(config.model_kwargs)
    kwargs.setdefault("teacher_forcing", True)
    return TransformerSeq2Seq(config, **kwargs).to(config.device, dtype=config.dtype)


def _build_cnn(config: BaselineConfig) -> nn.Module:
    model_kwargs = dict(config.model_kwargs)
    if "pad_token_id" not in model_kwargs:
        model_kwargs["pad_token_id"] = config.pad_token_id
    return CNN1DSeq2Seq(
        config.input_vocab_size,
        config.output_vocab_size,
        **model_kwargs,
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


def _build_nca1d(config: BaselineConfig) -> nn.Module:
    model = NCA1DSeq2Seq(config, **dict(config.model_kwargs))
    return model.to(config.device, dtype=config.dtype)


def _build_rnn_ar(config: BaselineConfig) -> nn.Module:
    kwargs = dict(config.model_kwargs)
    kwargs.setdefault("teacher_forcing", True)
    return RNNSeq2SeqAutoregressive(config, **kwargs).to(config.device, dtype=config.dtype)


BASELINE_REGISTRY: Dict[str, Callable[[BaselineConfig], nn.Module]] = {
    "rnn": _build_rnn,
    "transformer": _build_transformer,
    "transformer_ar": _build_transformer_ar,
    "cnn1d": _build_cnn,
    "tiny_recursive": _build_tiny_recursive,
    "trm": _build_tiny_recursive,
    "hrm": _build_hierarchical_recursive,
    "transformer_act": _build_transformer_act,
    "nca1d": _build_nca1d,
    "rnn_ar": _build_rnn_ar,
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
    "RNNSeq2SeqAutoregressive",
    "TinyRecursiveSeq2Seq",
    "HierarchicalReasoningSeq2Seq",
    "TransformerACTSeq2Seq",
    "TransformerSeq2Seq",
    "TransformerEncoderSeq2Seq",
    "NCA1DSeq2Seq",
    "create_baseline",
    "get_baseline_registry",
]
