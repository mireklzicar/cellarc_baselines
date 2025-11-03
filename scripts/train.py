#!/usr/bin/env python3
"""Training pipeline for Cell ARC baselines with Hydra configuration."""

from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import json
import shutil

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines import BaselineConfig, create_baseline, get_baseline_registry
from cellarc import EpisodeDataset


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
        shuffle: bool = False,
        shuffle_buffer: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._tokens = tokens
        self._max_seq_len = int(max_seq_len)
        self._drop_long = bool(drop_long)
        self._ignore_index = ignore_index
        self._shuffle = bool(shuffle)
        self._shuffle_buffer = (
            max(1, int(shuffle_buffer)) if shuffle and shuffle_buffer is not None else None
        )
        self._shuffle_seed = seed
        self._shuffle_iter = 0

    def __iter__(self) -> Iterator[EpisodeSequenceSample]:
        if not self._shuffle:
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
            return

        buffer_size = self._shuffle_buffer or 256
        base_seed = self._shuffle_seed if self._shuffle_seed is not None else random.randrange(2**31)
        rng = random.Random(base_seed + self._shuffle_iter)
        self._shuffle_iter += 1

        buffer: List[EpisodeSequenceSample] = []
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
            buffer.append(sample)
            if len(buffer) >= buffer_size:
                index = rng.randrange(len(buffer))
                yield buffer.pop(index)

        while buffer:
            index = rng.randrange(len(buffer))
            yield buffer.pop(index)


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


def init_wandb(cfg: DictConfig) -> Tuple[Optional[Any], Optional[Any]]:
    """Initialise a Weights & Biases run if enabled in the configuration."""

    logging_cfg = cfg.get("logging", None)
    if logging_cfg is None:
        return None, None
    wandb_cfg = logging_cfg.get("wandb", None)
    if not wandb_cfg or not wandb_cfg.get("enabled", False):
        return None, None

    try:
        import wandb  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Weights & Biases logging is enabled but the 'wandb' package is not installed. "
            "Install it with `pip install wandb` or disable logging.wandb.enabled."
        ) from exc

    tags = wandb_cfg.get("tags")
    if tags is not None:
        tags = list(tags)

    architecture = str(cfg.model.architecture)
    size_name = ""
    if cfg.model.size and "size" in cfg.model.size:
        size_name = str(cfg.model.size.size)

    auto_name = f"{architecture}_{size_name}".strip("_")
    specified_name = wandb_cfg.get("name")
    run_name = specified_name or auto_name

    run = wandb.init(
        project=wandb_cfg.get("project") or None,
        entity=wandb_cfg.get("entity") or None,
        name=run_name,
        group=wandb_cfg.get("group") or None,
        tags=tags,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    return run, wandb


def _sanitize_path_component(value: str) -> str:
    """Convert an arbitrary string into a filesystem-friendly component."""
    cleaned = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip()
    )
    cleaned = cleaned.strip("._")
    return cleaned or "default"


def resolve_checkpoint_directory(
    *,
    checkpoint_cfg: DictConfig,
    wandb_cfg: Optional[DictConfig],
    wandb_run: Optional[Any],
) -> Optional[Path]:
    """Determine the checkpoint directory inside the Hydra outputs tree."""

    base_outputs = Path(get_original_cwd()) / "outputs"
    base_outputs.mkdir(parents=True, exist_ok=True)

    use_wandb = (
        wandb_cfg is not None
        and bool(wandb_cfg.get("enabled", False))
        and wandb_run is not None
    )

    if use_wandb:
        project = wandb_cfg.get("project") or getattr(wandb_run, "project", None) or "wandb"
        run_name = (
            getattr(wandb_run, "name", None)
            or wandb_cfg.get("name")
            or getattr(wandb_run, "id", None)
        )
        if not run_name:
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        subdir = Path(_sanitize_path_component(project)) / _sanitize_path_component(run_name)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        subdir = Path(_sanitize_path_component(timestamp))

    checkpoint_dir = base_outputs / subdir
    if checkpoint_dir.exists() and not bool(checkpoint_cfg.get("overwrite", False)):
        prompt = f"Checkpoint directory '{checkpoint_dir}' already exists. Overwrite? [y/N]: "
        try:
            response = input(prompt)
        except EOFError:
            LOGGER.warning(
                "No interactive input available to confirm overwrite of %s. "
                "Disabling checkpoint saving.",
                checkpoint_dir,
            )
            return None
        if response.strip().lower() not in {"y", "yes"}:
            LOGGER.info(
                "User declined to overwrite existing checkpoint directory %s. "
                "Checkpoints will not be saved.",
                checkpoint_dir,
            )
            return None

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@dataclass
class CheckpointManager:
    """Handle saving last/best neural baseline checkpoints and evaluation summaries."""

    directory: Path
    best_accuracy: float = field(default_factory=lambda: float("-inf"))
    best_epoch: Optional[int] = None
    best_step: Optional[int] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_by_split: Dict[str, Dict[str, float]] = field(default_factory=dict)
    last_eval_metrics: Optional[Dict[str, float]] = None
    last_eval_by_split: Dict[str, Dict[str, float]] = field(default_factory=dict)
    last_epoch: Optional[int] = None
    last_step: Optional[int] = None
    _best_saved: bool = field(default=False, init=False)
    _last_path: Path = field(init=False)
    _best_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._last_path = self.directory / "model_last.pt"
        self._best_path = self.directory / "model_best.pt"

    @property
    def last_path(self) -> Path:
        return self._last_path

    @property
    def best_path(self) -> Path:
        return self._best_path

    def save_last(self, model: torch.nn.Module) -> None:
        """Persist the current model state as the latest checkpoint."""
        torch.save(model.state_dict(), self._last_path)

    def record_evaluation(
        self,
        *,
        model: torch.nn.Module,
        aggregated_metrics: Dict[str, float],
        metrics_by_split: Dict[str, Dict[str, float]],
        epoch: Optional[int],
        step: Optional[int],
    ) -> None:
        """Track evaluation results and update the best checkpoint if needed."""

        self.last_eval_metrics = dict(aggregated_metrics) if aggregated_metrics else None
        self.last_eval_by_split = {
            split: dict(values) for split, values in metrics_by_split.items()
        }
        self.last_epoch = epoch
        self.last_step = step

        if (
            epoch is not None
            and aggregated_metrics
            and "accuracy" in aggregated_metrics
        ):
            accuracy = float(aggregated_metrics["accuracy"])
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_epoch = epoch
                self.best_step = step
                self.best_metrics = dict(aggregated_metrics)
                self.best_by_split = {
                    split: dict(values) for split, values in metrics_by_split.items()
                }
                torch.save(model.state_dict(), self._best_path)
                self._best_saved = True
                LOGGER.info(
                    "New best checkpoint (accuracy=%.4f) at epoch %s step %s saved to %s",
                    accuracy,
                    epoch,
                    step,
                    self._best_path,
                )

    def finalize(
        self,
        *,
        total_steps: int,
        test_metrics_by_split: Dict[str, Dict[str, float]],
        test_aggregated_metrics: Dict[str, float],
    ) -> None:
        """Write evaluation summaries and ensure both checkpoint files exist."""

        if not self._best_saved and self._last_path.exists():
            shutil.copyfile(self._last_path, self._best_path)
            self._best_saved = True
            if self.best_metrics is None and self.last_eval_metrics is not None:
                self.best_metrics = dict(self.last_eval_metrics)
                self.best_by_split = {
                    split: dict(values) for split, values in self.last_eval_by_split.items()
                }
                accuracy = self.best_metrics.get("accuracy")
                if accuracy is not None:
                    self.best_accuracy = float(accuracy)
                self.best_epoch = self.last_epoch
                self.best_step = self.last_step
            LOGGER.info(
                "Copied model_last.pt to model_best.pt because no epoch-based best checkpoint was recorded."
            )

        summary: Dict[str, Any] = {
            "checkpoint_dir": str(self.directory),
            "artifacts": {
                "model_last": self._last_path.name,
                "model_best": self._best_path.name,
            },
            "train": {"steps": int(total_steps)},
        }

        if self.best_metrics is not None:
            accuracy = self.best_metrics.get("accuracy")
            summary["best_eval"] = {
                "epoch": self.best_epoch,
                "step": self.best_step,
                "aggregated": dict(self.best_metrics),
                "per_split": {
                    split: dict(values) for split, values in self.best_by_split.items()
                },
            }
            if accuracy is not None:
                summary["best_eval"]["accuracy"] = float(accuracy)

        if self.last_eval_metrics is not None:
            summary["last_eval"] = {
                "epoch": self.last_epoch,
                "step": self.last_step,
                "aggregated": dict(self.last_eval_metrics),
                "per_split": {
                    split: dict(values) for split, values in self.last_eval_by_split.items()
                },
            }

        if test_aggregated_metrics:
            summary["test"] = {
                "aggregated": dict(test_aggregated_metrics),
                "per_split": {
                    split: dict(values) for split, values in test_metrics_by_split.items()
                },
            }

        summary_path = self.directory / "evaluation.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        LOGGER.info("Wrote evaluation summary to %s", summary_path)


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


def log_training_metrics(
    *,
    epoch: int,
    step: int,
    lr: float,
    loss_sum: float,
    correct: int,
    tokens: int,
    wandb_module: Optional[Any],
) -> Dict[str, float]:
    """Log per-batch training metrics to the console and optionally to Weights & Biases."""

    if tokens <= 0:
        raise ValueError("Attempted to log training metrics without any batch tokens.")

    loss = loss_sum / tokens
    accuracy = correct / tokens if tokens else 0.0

    LOGGER.info(
        "epoch=%d step=%d lr=%.3e loss=%.4f acc=%.2f%% tokens=%d",
        epoch,
        step,
        lr,
        loss,
        accuracy * 100.0,
        int(tokens),
    )

    metrics = {
        "train/loss": float(loss),
        "train/accuracy": float(accuracy),
        "train/tokens": float(tokens),
        "train/lr": float(lr),
        "train/epoch": float(epoch),
    }
    if wandb_module:
        wandb_module.log(metrics, step=step)
    return metrics


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
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """Construct a DataLoader streaming flattened episode sequences."""

    shuffle_enabled = bool(cfg.data.get("shuffle", False)) if shuffle is None else bool(shuffle)
    shuffle_buffer: Optional[int] = None
    if shuffle_enabled:
        buffer_cfg = cfg.data.get("shuffle_buffer", None)
        if buffer_cfg is not None:
            shuffle_buffer = int(buffer_cfg)
        else:
            shuffle_buffer = max(batch_size * 4, 256)

    iterable = EpisodeSequenceDataset(
        dataset,
        tokens=tokens,
        max_seq_len=int(cfg.data.max_seq_len),
        drop_long=bool(cfg.data.drop_long_episodes),
        ignore_index=ignore_index,
        shuffle=shuffle_enabled,
        shuffle_buffer=shuffle_buffer,
        seed=int(cfg.seed),
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


def normalize_split_config(value: Any) -> List[str]:
    """Normalize split configuration entries into a list of strings."""

    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, ListConfig):
        return [str(item) for item in value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


def aggregate_split_metrics(
    metrics_by_split: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """Aggregate per-split metrics by averaging loss/accuracy and summing counts."""

    if not metrics_by_split:
        return {}

    aggregated: Dict[str, float] = {}
    for key in ("loss", "accuracy"):
        values = [metrics[key] for metrics in metrics_by_split.values() if key in metrics]
        if values:
            aggregated[key] = float(sum(values) / len(values))
    aggregated["episodes"] = float(
        sum(metrics.get("episodes", 0.0) for metrics in metrics_by_split.values())
    )
    aggregated["tokens"] = float(
        sum(metrics.get("tokens", 0.0) for metrics in metrics_by_split.values())
    )
    return aggregated


def evaluate_splits(
    model: torch.nn.Module,
    loaders: Dict[str, DataLoader],
    *,
    device: torch.device,
    ignore_index: int,
    max_episodes: Optional[int],
) -> Dict[str, Dict[str, float]]:
    """Evaluate the model over multiple splits and return metrics per split."""

    return {
        split_name: evaluate(
            model,
            loader,
            device=device,
            ignore_index=ignore_index,
            max_episodes=max_episodes,
        )
        for split_name, loader in loaders.items()
    }


def log_split_metrics(
    *,
    tag: str,
    metrics_by_split: Dict[str, Dict[str, float]],
    aggregated_metrics: Dict[str, float],
    step: int,
    epoch: Optional[int],
    wandb_module: Optional[Any],
) -> None:
    """Log per-split and aggregated metrics to the console and optionally to Weights & Biases."""

    epoch_str = f" epoch={epoch}" if epoch is not None else ""
    for split_name in sorted(metrics_by_split):
        metrics = metrics_by_split[split_name]
        LOGGER.info(
            "%s%s split=%s step=%d loss=%.4f acc=%.2f%% episodes=%d tokens=%d",
            tag,
            epoch_str,
            split_name,
            step,
            metrics.get("loss", 0.0),
            metrics.get("accuracy", 0.0) * 100.0,
            int(metrics.get("episodes", 0.0)),
            int(metrics.get("tokens", 0.0)),
        )
    if aggregated_metrics:
        LOGGER.info(
            "%s%s split=mean step=%d loss=%.4f acc=%.2f%% episodes=%d tokens=%d",
            tag,
            epoch_str,
            step,
            aggregated_metrics.get("loss", 0.0),
            aggregated_metrics.get("accuracy", 0.0) * 100.0,
            int(aggregated_metrics.get("episodes", 0.0)),
            int(aggregated_metrics.get("tokens", 0.0)),
        )

    if not wandb_module:
        return

    payload: Dict[str, float] = {}
    for split_name in metrics_by_split:
        metrics = metrics_by_split[split_name]
        payload[f"{tag}/loss/{split_name}"] = float(metrics.get("loss", 0.0))
        payload[f"{tag}/accuracy/{split_name}"] = float(metrics.get("accuracy", 0.0))
        payload[f"{tag}/episodes/{split_name}"] = float(metrics.get("episodes", 0.0))
    if aggregated_metrics:
        payload[f"{tag}/loss_mean"] = float(aggregated_metrics.get("loss", 0.0))
        payload[f"{tag}/accuracy_mean"] = float(aggregated_metrics.get("accuracy", 0.0))
        payload[f"{tag}/episodes_total"] = float(aggregated_metrics.get("episodes", 0.0))

    if payload:
        wandb_module.log(payload, step=step)


def instantiate_episode_dataset(
    cfg: DictConfig,
    split: str,
    seed: int,
    *,
    augmentation: Optional[Dict[str, Any]] = None,
) -> EpisodeDataset:
    """Instantiate an EpisodeDataset from the huggingface snapshots."""

    dataset_kwargs: Dict[str, Any] = {
        "split": split,
        "name": str(cfg.dataset.benchmark),
        "include_metadata": bool(cfg.dataset.include_metadata),
        "root": cfg.dataset.root or None,
        "seed": seed,
    }
    if augmentation and augmentation.get("enabled", False):
        dataset_kwargs.update(
            augment=True,
            reverse_prob=float(augmentation.get("reverse_prob", 0.5)),
            palette=list(augmentation.get("palette", [])),
        )
    else:
        dataset_kwargs["augment"] = False

    return EpisodeDataset.from_huggingface(**dataset_kwargs)


def resolve_augmentation_settings(
    cfg: DictConfig,
    tokens: TokenVocabulary,
) -> Dict[str, Any]:
    """Derive augmentation parameters from the configuration."""

    data_cfg = cfg.get("data", {})
    augmentation_cfg = data_cfg.get("augmentation", None)
    if not augmentation_cfg or not augmentation_cfg.get("enabled", False):
        return {"enabled": False}

    palette_cfg = augmentation_cfg.get("palette")
    if palette_cfg is None or (isinstance(palette_cfg, ListConfig) and len(palette_cfg) == 0):
        palette = list(range(tokens.num_colors))
    else:
        if isinstance(palette_cfg, ListConfig):
            palette_iterable = list(palette_cfg)
        elif isinstance(palette_cfg, Sequence):
            palette_iterable = list(palette_cfg)
        else:
            palette_iterable = [palette_cfg]
        palette = [int(value) for value in palette_iterable]

    return {
        "enabled": True,
        "reverse_prob": float(augmentation_cfg.get("reverse_prob", 0.5)),
        "palette": palette,
    }


@hydra.main(config_path="../configs", config_name="train/default", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    LOGGER.info("Working directory: %s", os.getcwd())
    LOGGER.info("Original project directory: %s", get_original_cwd())
    LOGGER.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    set_all_seeds(int(cfg.seed))

    wandb_run, wandb_module = init_wandb(cfg)
    logging_cfg = cfg.get("logging", None)
    wandb_cfg = logging_cfg.get("wandb", None) if logging_cfg else None

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

    augmentation_settings = resolve_augmentation_settings(cfg, tokens)
    seed = int(cfg.seed)

    train_split = str(cfg.dataset.train_split)
    train_dataset = instantiate_episode_dataset(
        cfg,
        train_split,
        seed=seed,
        augmentation=augmentation_settings,
    )
    train_loader = make_episode_dataloader(
        dataset=train_dataset,
        tokens=tokens,
        cfg=cfg,
        batch_size=int(cfg.data.batch_size),
        ignore_index=ignore_index,
        shuffle=bool(cfg.data.shuffle),
    )

    eval_batch_size = int(cfg.eval.batch_size)
    eval_split_cfg = cfg.dataset.get("eval_splits", None)
    if eval_split_cfg is None and cfg.dataset.get("eval_split", None):
        eval_split_cfg = cfg.dataset.eval_split
    eval_split_names = normalize_split_config(eval_split_cfg)
    eval_loaders: Dict[str, DataLoader] = {}
    for split_name in eval_split_names:
        eval_dataset = instantiate_episode_dataset(
            cfg,
            split_name,
            seed=seed,
            augmentation=None,
        )
        eval_loaders[split_name] = make_episode_dataloader(
            dataset=eval_dataset,
            tokens=tokens,
            cfg=cfg,
            batch_size=eval_batch_size,
            ignore_index=ignore_index,
            shuffle=False,
        )

    train_eval_split_cfg = cfg.dataset.get("train_eval_splits", None)
    train_eval_split_names = normalize_split_config(train_eval_split_cfg)
    for split_name in train_eval_split_names:
        if split_name in eval_loaders:
            continue
        try:
            eval_dataset = instantiate_episode_dataset(
                cfg,
                split_name,
                seed=seed,
                augmentation=None,
            )
        except Exception as exc:  # pragma: no cover - dataset availability depends on external data
            LOGGER.warning(
                "Skipping train evaluation split '%s' due to error: %s", split_name, exc
            )
            continue
        eval_loaders[split_name] = make_episode_dataloader(
            dataset=eval_dataset,
            tokens=tokens,
            cfg=cfg,
            batch_size=eval_batch_size,
            ignore_index=ignore_index,
            shuffle=False,
        )

    test_cfg = cfg.get("test", None)
    test_split_cfg = cfg.dataset.get("test_splits", None)
    if test_split_cfg is None and cfg.dataset.get("test_split", None):
        test_split_cfg = cfg.dataset.test_split
    test_split_names = normalize_split_config(test_split_cfg)
    test_loaders: Dict[str, DataLoader] = {}
    test_limit = None
    if test_cfg is not None and test_split_names:
        test_batch_size = int(test_cfg.get("batch_size", cfg.eval.batch_size))
        for split_name in test_split_names:
            test_dataset = instantiate_episode_dataset(
                cfg,
                split_name,
                seed=seed,
                augmentation=None,
            )
            test_loaders[split_name] = make_episode_dataloader(
                dataset=test_dataset,
                tokens=tokens,
                cfg=cfg,
                batch_size=test_batch_size,
                ignore_index=ignore_index,
                shuffle=False,
            )
        if test_cfg.get("limit") is not None:
            test_limit = int(test_cfg.get("limit"))

    model, device = prepare_baseline_model(cfg, tokens)
    model.to(device)

    if wandb_module and wandb_run:
        num_parameters = sum(param.numel() for param in model.parameters())
        wandb_run.summary["model/num_parameters"] = int(num_parameters)

    if wandb_module and wandb_run and wandb_cfg and wandb_cfg.get("log_model", False):
        wandb_module.watch(model, log="all", log_freq=max(1, int(cfg.trainer.log_every)))

    checkpoint_manager: Optional[CheckpointManager] = None
    checkpoint_cfg = cfg.trainer.get("checkpoints", None)
    if checkpoint_cfg and bool(checkpoint_cfg.get("enabled", False)):
        if not isinstance(model, torch.nn.Module):
            LOGGER.warning(
                "Checkpoint saving is only supported for neural baselines. Skipping request."
            )
        else:
            checkpoint_dir = resolve_checkpoint_directory(
                checkpoint_cfg=checkpoint_cfg,
                wandb_cfg=wandb_cfg,
                wandb_run=wandb_run,
            )
            if checkpoint_dir is not None:
                checkpoint_manager = CheckpointManager(checkpoint_dir)
                LOGGER.info(
                    "Checkpointing enabled; artifacts will be saved under %s",
                    checkpoint_dir,
                )
            else:
                LOGGER.warning("Checkpointing disabled because the directory could not be prepared.")

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

    gradient_accumulation = max(1, int(cfg.trainer.gradient_accumulation))
    clip_grad_norm = cfg.trainer.clip_grad_norm

    num_epochs_cfg = cfg.trainer.get("num_epochs", None)
    num_epochs = int(num_epochs_cfg) if num_epochs_cfg is not None else None
    if num_epochs is not None and num_epochs <= 0:
        num_epochs = None

    max_steps_cfg = cfg.trainer.get("max_steps", None)
    max_steps = int(max_steps_cfg) if max_steps_cfg is not None else None
    if max_steps is not None and max_steps <= 0:
        max_steps = None

    eval_every_cfg = cfg.trainer.get("eval_every", None)
    eval_every = int(eval_every_cfg) if eval_every_cfg is not None else None
    if eval_every is not None and eval_every <= 0:
        eval_every = None

    log_every_cfg = cfg.trainer.get("log_every", None)
    if log_every_cfg is None:
        raise ValueError("trainer.log_every must be provided.")
    log_every = int(log_every_cfg)
    if log_every <= 0:
        raise ValueError("trainer.log_every must be positive.")

    max_eval_episodes_cfg = cfg.trainer.get("max_eval_episodes", None)
    max_eval_episodes = (
        int(max_eval_episodes_cfg) if max_eval_episodes_cfg is not None else None
    )
    eval_limit_cfg = cfg.eval.get("limit", None)
    if eval_limit_cfg is not None:
        eval_limit = int(eval_limit_cfg)
        if max_eval_episodes is None:
            max_eval_episodes = eval_limit
        else:
            max_eval_episodes = min(max_eval_episodes, eval_limit)

    use_epoch_training = num_epochs is not None
    if not use_epoch_training and max_steps is None:
        raise ValueError(
            "Specify either trainer.num_epochs or trainer.max_steps to terminate training."
        )

    latest_eval_metrics: Optional[Dict[str, float]] = None
    latest_eval_by_split: Dict[str, Dict[str, float]] = {}
    last_train_metrics: Optional[Dict[str, float]] = None

    LOGGER.info("Starting training on %s", device)
    global_step = 0
    current_epoch = 0
    should_stop = False
    last_step_stats: Optional[Dict[str, Any]] = None
    last_logged_step = -1

    while not should_stop:
        current_epoch += 1
        if use_epoch_training and num_epochs is not None:
            LOGGER.info("Starting epoch %d/%d", current_epoch, num_epochs)
        else:
            LOGGER.info("Starting epoch %d", current_epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        step_correct = 0
        step_tokens = 0
        micro_batches = 0

        for batch in train_loader:
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
            micro_batches += 1

            if micro_batches == gradient_accumulation:
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(clip_grad_norm)
                    )

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                global_step += 1

                current_lr = optimizer.param_groups[0]["lr"]
                last_step_stats = {
                    "epoch": current_epoch,
                    "step": global_step,
                    "lr": float(current_lr),
                    "loss_sum": float(step_loss),
                    "correct": int(step_correct),
                    "tokens": int(step_tokens),
                }

                if step_tokens > 0 and global_step % log_every == 0:
                    last_train_metrics = log_training_metrics(
                        epoch=current_epoch,
                        step=global_step,
                        lr=float(current_lr),
                        loss_sum=float(step_loss),
                        correct=int(step_correct),
                        tokens=int(step_tokens),
                        wandb_module=wandb_module,
                    )
                    last_logged_step = global_step

                if (
                    eval_loaders
                    and not use_epoch_training
                    and eval_every
                    and global_step % eval_every == 0
                ):
                    metrics_by_split = evaluate_splits(
                        model,
                        eval_loaders,
                        device=device,
                        ignore_index=ignore_index,
                        max_episodes=max_eval_episodes,
                    )
                    aggregated_metrics = aggregate_split_metrics(metrics_by_split)
                    latest_eval_metrics = aggregated_metrics
                    latest_eval_by_split = metrics_by_split
                    log_split_metrics(
                        tag="val",
                        metrics_by_split=metrics_by_split,
                        aggregated_metrics=aggregated_metrics,
                        step=global_step,
                        epoch=None,
                        wandb_module=wandb_module,
                    )
                    if checkpoint_manager:
                        checkpoint_manager.record_evaluation(
                            model=model,
                            aggregated_metrics=aggregated_metrics,
                            metrics_by_split=metrics_by_split,
                            epoch=None,
                            step=global_step,
                        )

                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                step_correct = 0
                step_tokens = 0
                micro_batches = 0

                if max_steps is not None and global_step >= max_steps:
                    break

        if micro_batches > 0 and (max_steps is None or global_step < max_steps):
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad_norm))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

            current_lr = optimizer.param_groups[0]["lr"]
            last_step_stats = {
                "epoch": current_epoch,
                "step": global_step,
                "lr": float(current_lr),
                "loss_sum": float(step_loss),
                "correct": int(step_correct),
                "tokens": int(step_tokens),
            }

            if step_tokens > 0 and global_step % log_every == 0:
                last_train_metrics = log_training_metrics(
                    epoch=current_epoch,
                    step=global_step,
                    lr=float(current_lr),
                    loss_sum=float(step_loss),
                    correct=int(step_correct),
                    tokens=int(step_tokens),
                    wandb_module=wandb_module,
                )
                last_logged_step = global_step

        if (
            eval_loaders
            and not use_epoch_training
            and eval_every
            and global_step % eval_every == 0
        ):
            metrics_by_split = evaluate_splits(
                model,
                eval_loaders,
                device=device,
                ignore_index=ignore_index,
                max_episodes=max_eval_episodes,
            )
            aggregated_metrics = aggregate_split_metrics(metrics_by_split)
            latest_eval_metrics = aggregated_metrics
            latest_eval_by_split = metrics_by_split
            log_split_metrics(
                tag="val",
                metrics_by_split=metrics_by_split,
                aggregated_metrics=aggregated_metrics,
                step=global_step,
                epoch=None,
                wandb_module=wandb_module,
            )
            if checkpoint_manager:
                checkpoint_manager.record_evaluation(
                    model=model,
                    aggregated_metrics=aggregated_metrics,
                    metrics_by_split=metrics_by_split,
                    epoch=None,
                    step=global_step,
                )

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_correct = 0
        step_tokens = 0
        micro_batches = 0

        if max_steps is not None and global_step >= max_steps:
            should_stop = True

        run_epoch_eval = (
            use_epoch_training
            and bool(eval_loaders)
            and not (max_steps is not None and global_step >= max_steps)
        )
        if run_epoch_eval:
            metrics_by_split = evaluate_splits(
                model,
                eval_loaders,
                device=device,
                ignore_index=ignore_index,
                max_episodes=max_eval_episodes,
            )
            aggregated_metrics = aggregate_split_metrics(metrics_by_split)
            latest_eval_metrics = aggregated_metrics
            latest_eval_by_split = metrics_by_split
            log_split_metrics(
                tag="val",
                metrics_by_split=metrics_by_split,
                aggregated_metrics=aggregated_metrics,
                step=global_step,
                epoch=current_epoch,
                wandb_module=wandb_module,
            )
            if checkpoint_manager:
                checkpoint_manager.record_evaluation(
                    model=model,
                    aggregated_metrics=aggregated_metrics,
                    metrics_by_split=metrics_by_split,
                    epoch=current_epoch,
                    step=global_step,
                )

        if use_epoch_training and num_epochs is not None and current_epoch >= num_epochs:
            should_stop = True

        if not use_epoch_training and max_steps is not None and global_step >= max_steps:
            should_stop = True

        if checkpoint_manager:
            checkpoint_manager.save_last(model)

        if should_stop:
            break

    if checkpoint_manager:
        checkpoint_manager.save_last(model)

    if (
        last_step_stats
        and int(last_step_stats["tokens"]) > 0
        and int(last_step_stats["step"]) != last_logged_step
    ):
        last_train_metrics = log_training_metrics(
            epoch=int(last_step_stats["epoch"]),
            step=int(last_step_stats["step"]),
            lr=float(last_step_stats["lr"]),
            loss_sum=float(last_step_stats["loss_sum"]),
            correct=int(last_step_stats["correct"]),
            tokens=int(last_step_stats["tokens"]),
            wandb_module=wandb_module,
        )
        last_logged_step = int(last_step_stats["step"])

    LOGGER.info("Training complete. Ran for %d optimisation steps.", global_step)

    test_metrics_by_split: Dict[str, Dict[str, float]] = {}
    test_aggregated_metrics: Dict[str, float] = {}
    if test_loaders:
        test_metrics_by_split = evaluate_splits(
            model,
            test_loaders,
            device=device,
            ignore_index=ignore_index,
            max_episodes=test_limit,
        )
        test_aggregated_metrics = aggregate_split_metrics(test_metrics_by_split)
        log_split_metrics(
            tag="test",
            metrics_by_split=test_metrics_by_split,
            aggregated_metrics=test_aggregated_metrics,
            step=global_step,
            epoch=None,
            wandb_module=wandb_module,
        )

    if checkpoint_manager:
        checkpoint_manager.finalize(
            total_steps=global_step,
            test_metrics_by_split=test_metrics_by_split,
            test_aggregated_metrics=test_aggregated_metrics,
        )

    if wandb_run:
        wandb_run.summary["train/steps"] = global_step
        if last_train_metrics:
            for key, value in last_train_metrics.items():
                wandb_run.summary[key] = value
        if latest_eval_metrics:
            wandb_run.summary["val/loss_mean"] = latest_eval_metrics.get("loss")
            wandb_run.summary["val/accuracy_mean"] = latest_eval_metrics.get("accuracy")
        for split_name, metrics in latest_eval_by_split.items():
            wandb_run.summary[f"val/{split_name}/loss"] = metrics.get("loss")
            wandb_run.summary[f"val/{split_name}/accuracy"] = metrics.get("accuracy")
        if test_aggregated_metrics:
            wandb_run.summary["test/loss_mean"] = test_aggregated_metrics.get("loss")
            wandb_run.summary["test/accuracy_mean"] = test_aggregated_metrics.get(
                "accuracy"
            )
        for split_name, metrics in test_metrics_by_split.items():
            wandb_run.summary[f"test/{split_name}/loss"] = metrics.get("loss")
            wandb_run.summary[f"test/{split_name}/accuracy"] = metrics.get("accuracy")
        wandb_run.finish()


if __name__ == "__main__":
    main()
