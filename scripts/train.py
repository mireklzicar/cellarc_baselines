#!/usr/bin/env python3
"""Training pipeline for Cell ARC baselines with Hydra configuration."""

from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines import BaselineConfig, create_baseline, get_baseline_registry
from cell_arc import EpisodeDataset


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenVocabulary:
    """Token identifiers used to linearise Cell ARC episodes."""

    num_colors: int
    pad_id: int
    support_input_id: int
    support_output_id: int
    query_id: int
    target_id: int
    mask_id: int
    eos_id: int

    def __post_init__(self) -> None:
        special = [
            self.pad_id,
            self.support_input_id,
            self.support_output_id,
            self.query_id,
            self.target_id,
            self.mask_id,
            self.eos_id,
        ]
        if min(special) < self.num_colors:
            raise ValueError(
                "Special token ids must not collide with colour tokens (0..num_colors-1)."
            )
        if len(set(special)) != len(special):
            raise ValueError("Special token ids must be unique.")

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size implied by the colour and special tokens."""
        return max(
            self.num_colors - 1,
            self.pad_id,
            self.support_input_id,
            self.support_output_id,
            self.query_id,
            self.target_id,
            self.mask_id,
            self.eos_id,
        ) + 1


@dataclass
class EpisodeSequenceSample:
    """Flattened representation of a single Cell ARC episode."""

    input_ids: List[int]
    target_ids: List[int]
    labels: List[int]
    loss_mask: List[bool]


def flatten_episode(
    episode: Dict[str, object],
    *,
    tokens: TokenVocabulary,
    max_seq_len: int,
    drop_long: bool,
    ignore_index: int,
) -> Optional[EpisodeSequenceSample]:
    """Convert an ARC episode into a flattened in-context sequence."""

    train_pairs = episode.get("train") or []
    query = episode.get("query") or []
    solution = episode.get("solution") or []

    sample = EpisodeSequenceSample([], [], [], [])

    def append_token(
        token_id: int,
        *,
        target_id: Optional[int] = None,
        label: Optional[int] = None,
        contributes_to_loss: bool = False,
    ) -> None:
        sample.input_ids.append(int(token_id))
        sample.target_ids.append(
            int(target_id) if target_id is not None else tokens.pad_id
        )
        if label is None:
            sample.labels.append(ignore_index)
        else:
            sample.labels.append(int(label))
        sample.loss_mask.append(bool(contributes_to_loss))

    for pair in train_pairs:
        append_token(tokens.support_input_id)
        for value in pair["input"]:
            append_token(int(value))
        append_token(tokens.support_output_id)
        for value in pair["output"]:
            append_token(int(value))

    append_token(tokens.query_id)
    for value in query:
        append_token(int(value))

    append_token(tokens.target_id, target_id=tokens.target_id)
    for value in solution:
        colour = int(value)
        append_token(
            tokens.mask_id,
            target_id=colour,
            label=colour,
            contributes_to_loss=True,
        )

    append_token(tokens.eos_id)

    total_length = len(sample.input_ids)
    if total_length == 0:
        return None

    if total_length > max_seq_len:
        if drop_long:
            return None
        truncated = max_seq_len
        sample.input_ids = sample.input_ids[:truncated]
        sample.target_ids = sample.target_ids[:truncated]
        sample.labels = sample.labels[:truncated]
        sample.loss_mask = sample.loss_mask[:truncated]

    return sample


class EpisodeSequenceDataset(IterableDataset):
    """PyTorch iterable dataset that streams flattened Cell ARC episodes."""

    def __init__(
        self,
        dataset: EpisodeDataset,
        *,
        tokens: TokenVocabulary,
        max_seq_len: int,
        drop_long: bool,
        ignore_index: int,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._tokens = tokens
        self._max_seq_len = int(max_seq_len)
        self._drop_long = bool(drop_long)
        self._ignore_index = ignore_index

    def __iter__(self) -> Iterator[EpisodeSequenceSample]:
        for episode in self._dataset:
            sample = flatten_episode(
                episode,
                tokens=self._tokens,
                max_seq_len=self._max_seq_len,
                drop_long=self._drop_long,
                ignore_index=self._ignore_index,
            )
            if sample is None:
                continue
            yield sample


def build_collate_fn(
    tokens: TokenVocabulary, ignore_index: int, pad_to_length: Optional[int] = None
):
    """Create a collate function that pads sequences to uniform length."""

    pad_token = tokens.pad_id

    def collate(samples: List[EpisodeSequenceSample]) -> Dict[str, torch.Tensor]:
        if not samples:
            raise ValueError("Received an empty batch.")
        max_len = max(len(sample.input_ids) for sample in samples)
        if pad_to_length is not None:
            if pad_to_length < max_len:
                raise ValueError(
                    f"pad_to_length={pad_to_length} is smaller than the longest sample ({max_len})."
                )
            max_len = pad_to_length
        batch_size = len(samples)

        inputs = torch.full((batch_size, max_len), pad_token, dtype=torch.long)
        targets = torch.full((batch_size, max_len), pad_token, dtype=torch.long)
        labels = torch.full((batch_size, max_len), ignore_index, dtype=torch.long)
        loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        lengths = torch.zeros(batch_size, dtype=torch.long)

        for idx, sample in enumerate(samples):
            length = len(sample.input_ids)
            lengths[idx] = length
            inputs[idx, :length] = torch.tensor(sample.input_ids, dtype=torch.long)
            targets[idx, :length] = torch.tensor(sample.target_ids, dtype=torch.long)
            labels[idx, :length] = torch.tensor(sample.labels, dtype=torch.long)
            loss_mask[idx, :length] = torch.tensor(sample.loss_mask, dtype=torch.bool)

        return {
            "inputs": inputs,
            "targets": targets,
            "labels": labels,
            "loss_mask": loss_mask,
            "lengths": lengths,
        }

    return collate


def infinite_loader(loader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    """Yield batches from a DataLoader indefinitely."""

    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)


def detect_device(device_cfg: str) -> torch.device:
    """Resolve the desired training device."""

    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # pragma: no cover - macOS only
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


def set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_baseline_model(
    cfg: DictConfig, tokens: TokenVocabulary
) -> tuple[torch.nn.Module, torch.device]:
    """Instantiate a baseline model according to the Hydra configuration."""

    architecture = str(cfg.model.architecture)
    registry = get_baseline_registry()
    if architecture not in registry:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available baselines: {sorted(registry)}"
        )

    dtype_name = str(cfg.model.dtype)
    try:
        dtype = getattr(torch, dtype_name)
    except AttributeError as exc:  # pragma: no cover - config validation
        raise ValueError(f"Unsupported dtype '{dtype_name}' in model config.") from exc

    requested_device = str(cfg.model.device)
    device = detect_device(requested_device)
    cpu_only_architectures = {"tiny_recursive", "trm", "hrm"}
    if architecture in cpu_only_architectures and device.type != "cpu":
        LOGGER.warning(
            "Architecture '%s' only supports CPU execution. Overriding requested device '%s' with CPU.",
            architecture,
            requested_device,
        )
        device = torch.device("cpu")

    size_cfg = cfg.model.size
    model_kwargs = {}
    if size_cfg and "variants" in size_cfg and architecture in size_cfg.variants:
        model_kwargs = OmegaConf.to_container(size_cfg.variants[architecture], resolve=True)  # type: ignore[assignment]

    baseline_config = BaselineConfig(
        input_vocab_size=tokens.vocab_size,
        output_vocab_size=tokens.vocab_size,
        max_seq_len=int(cfg.data.max_seq_len),
        batch_size=int(cfg.data.batch_size),
        device=device,
        dtype=dtype,
        model_kwargs=model_kwargs or {},
    )
    model = create_baseline(architecture, baseline_config)
    return model, device


def compute_batch_loss(
    logits: torch.Tensor,
    *,
    labels: torch.Tensor,
    ignore_index: int,
) -> Dict[str, torch.Tensor]:
    """Compute loss and accuracy statistics for a batch."""

    vocab_size = logits.size(-1)
    reshaped_logits = logits.reshape(-1, vocab_size)
    reshaped_labels = labels.view(-1)
    valid_mask = reshaped_labels != ignore_index
    valid_count = valid_mask.long().sum()

    if valid_count.item() == 0:
        return {
            "loss": logits.new_tensor(0.0),
            "correct": logits.new_tensor(0),
            "total": logits.new_tensor(0),
        }

    loss = torch.nn.functional.cross_entropy(
        reshaped_logits, reshaped_labels, ignore_index=ignore_index
    )
    predictions = reshaped_logits.argmax(dim=-1)
    correct = (predictions == reshaped_labels) & valid_mask

    return {
        "loss": loss,
        "correct": correct.long().sum(),
        "total": valid_count,
    }


def forward_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Forward pass abstraction handling optional target tensors."""

    inputs = batch["inputs"]
    targets = batch["targets"]
    if getattr(model, "requires_targets", False):
        return model(inputs, targets=targets)
    return model(inputs)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    ignore_index: int,
    max_episodes: Optional[int],
) -> Dict[str, float]:
    """Run evaluation over the provided dataloader."""

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_episodes = 0

    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }
            logits = forward_batch(model, batch)
            stats = compute_batch_loss(
                logits,
                labels=batch["labels"],
                ignore_index=ignore_index,
            )

            valid_tokens = int(stats["total"].item())
            if valid_tokens > 0:
                total_loss += float(stats["loss"].item()) * valid_tokens
                total_tokens += valid_tokens
                total_correct += int(stats["correct"].item())

            total_episodes += batch["inputs"].size(0)
            if max_episodes is not None and total_episodes >= max_episodes:
                break

    model.train()

    mean_loss = total_loss / total_tokens if total_tokens else 0.0
    accuracy = total_correct / total_tokens if total_tokens else 0.0

    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "episodes": float(total_episodes),
        "tokens": float(total_tokens),
    }


def make_episode_dataloader(
    *,
    dataset: EpisodeDataset,
    tokens: TokenVocabulary,
    cfg: DictConfig,
    batch_size: int,
    ignore_index: int,
) -> DataLoader:
    """Construct a DataLoader streaming flattened episode sequences."""

    iterable = EpisodeSequenceDataset(
        dataset,
        tokens=tokens,
        max_seq_len=int(cfg.data.max_seq_len),
        drop_long=bool(cfg.data.drop_long_episodes),
        ignore_index=ignore_index,
    )
    collate_fn = build_collate_fn(
        tokens,
        ignore_index,
        pad_to_length=int(cfg.data.max_seq_len),
    )
    loader = DataLoader(
        iterable,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        collate_fn=collate_fn,
    )
    return loader


def instantiate_episode_dataset(cfg: DictConfig, split: str, seed: int) -> EpisodeDataset:
    """Instantiate an EpisodeDataset from the huggingface snapshots."""

    return EpisodeDataset.from_huggingface(
        split=split,
        name=str(cfg.dataset.benchmark),
        include_metadata=bool(cfg.dataset.include_metadata),
        root=cfg.dataset.root,
        seed=seed,
    )


@hydra.main(config_path="../configs", config_name="train/default", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    LOGGER.info("Working directory: %s", os.getcwd())
    LOGGER.info("Original project directory: %s", get_original_cwd())
    LOGGER.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    set_all_seeds(int(cfg.seed))

    tokens = TokenVocabulary(
        num_colors=int(cfg.tokens.num_colors),
        pad_id=int(cfg.tokens.pad_id),
        support_input_id=int(cfg.tokens.support_input_id),
        support_output_id=int(cfg.tokens.support_output_id),
        query_id=int(cfg.tokens.query_id),
        target_id=int(cfg.tokens.target_id),
        mask_id=int(cfg.tokens.mask_id),
        eos_id=int(cfg.tokens.eos_id),
    )
    ignore_index = int(cfg.loss.ignore_index)

    if cfg.data.shuffle:
        LOGGER.warning(
            "data.shuffle=True requested but is not yet implemented for the streaming loader; "
            "continuing without shuffling."
        )

    train_dataset = instantiate_episode_dataset(cfg, cfg.dataset.train_split, seed=int(cfg.seed))
    eval_dataset = None
    if cfg.dataset.eval_split:
        eval_dataset = instantiate_episode_dataset(cfg, cfg.dataset.eval_split, seed=int(cfg.seed))

    train_loader = make_episode_dataloader(
        dataset=train_dataset,
        tokens=tokens,
        cfg=cfg,
        batch_size=int(cfg.data.batch_size),
        ignore_index=ignore_index,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = make_episode_dataloader(
            dataset=eval_dataset,
            tokens=tokens,
            cfg=cfg,
            batch_size=int(cfg.eval.batch_size),
            ignore_index=ignore_index,
        )

    model, device = prepare_baseline_model(cfg, tokens)
    model.to(device)

    optimizer_name = str(cfg.optimizer.name).lower()
    if optimizer_name != "adamw":  # pragma: no cover - config validation
        raise ValueError(f"Unsupported optimizer '{cfg.optimizer.name}'. Only AdamW is implemented.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optimizer.lr),
        betas=tuple(cfg.optimizer.betas) if "betas" in cfg.optimizer else (0.9, 0.999),
        eps=float(cfg.optimizer.eps),
        weight_decay=float(cfg.optimizer.weight_decay),
    )

    scheduler = None
    warmup_steps = int(cfg.optimizer.warmup_steps)
    if warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0),
        )

    loader_iter = infinite_loader(train_loader)
    gradient_accumulation = max(1, int(cfg.trainer.gradient_accumulation))
    clip_grad_norm = cfg.trainer.clip_grad_norm

    max_steps = int(cfg.trainer.max_steps)
    eval_every = int(cfg.trainer.eval_every)
    log_every = int(cfg.trainer.log_every)
    max_eval_episodes = (
        int(cfg.trainer.max_eval_episodes) if cfg.trainer.max_eval_episodes is not None else None
    )
    eval_limit = None
    if cfg.eval.limit is not None:
        eval_limit = int(cfg.eval.limit)
        if max_eval_episodes is None:
            max_eval_episodes = eval_limit
        else:
            max_eval_episodes = min(max_eval_episodes, eval_limit)

    LOGGER.info("Starting training on %s", device)
    global_step = 0
    while global_step < max_steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        step_correct = 0
        step_tokens = 0

        for _ in range(gradient_accumulation):
            batch = next(loader_iter)
            batch = {key: value.to(device) for key, value in batch.items()}

            logits = forward_batch(model, batch)
            stats = compute_batch_loss(
                logits,
                labels=batch["labels"],
                ignore_index=ignore_index,
            )
            valid_tokens = int(stats["total"].item())
            if valid_tokens == 0:
                continue

            loss = stats["loss"] / gradient_accumulation
            loss.backward()

            step_loss += float(stats["loss"].item()) * valid_tokens
            step_correct += int(stats["correct"].item())
            step_tokens += valid_tokens

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad_norm))

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        global_step += 1

        if step_tokens > 0 and global_step % log_every == 0:
            avg_loss = step_loss / step_tokens
            accuracy = step_correct / step_tokens if step_tokens else 0.0
            current_lr = optimizer.param_groups[0]["lr"]
            LOGGER.info(
                "step=%d lr=%.3e loss=%.4f acc=%.2f%% tokens=%d",
                global_step,
                current_lr,
                avg_loss,
                accuracy * 100.0,
                step_tokens,
            )

        if eval_loader is not None and eval_every > 0 and global_step % eval_every == 0:
            metrics = evaluate(
                model,
                eval_loader,
                device=device,
                ignore_index=ignore_index,
                max_episodes=max_eval_episodes,
            )
            LOGGER.info(
                "eval step=%d loss=%.4f acc=%.2f%% episodes=%d tokens=%d",
                global_step,
                metrics["loss"],
                metrics["accuracy"] * 100.0,
                int(metrics["episodes"]),
                int(metrics["tokens"]),
            )

    LOGGER.info("Training complete. Ran for %d optimisation steps.", global_step)


if __name__ == "__main__":
    main()
