#!/usr/bin/env python3
"""Training pipeline for puzzle-embedding (single I/O) supervision."""

from __future__ import annotations

import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from baselines import BaselineConfig, create_baseline, get_baseline_registry
from baselines.neural.recursive_reasoning.utils.sparse_embedding import (
    CastedSparseEmbedding,
)
from scripts.train_incontext import (
    TokenVocabulary,
    aggregate_split_metrics,
    CheckpointManager,
    compute_batch_loss,
    detect_device,
    init_wandb,
    instantiate_episode_dataset,
    log_split_metrics,
    log_training_metrics,
    normalize_split_config,
    resolve_augmentation_settings,
    resolve_checkpoint_directory,
    sample_palette_perm,
    invert_perm,
    apply_palette,
    reverse_value_segments,
    set_all_seeds,
)
from scripts.training.optimizers import AdamATan2
from scripts.training.puzzle_embedding_dataset import (
    PuzzleEmbeddingIterableDataset,
    PuzzleIdentifierTable,
    build_puzzle_embedding_collate_fn,
)
from scripts.training.recursive_utils import EMAHelper, PuzzleEmbeddingOptimizer

LOGGER = logging.getLogger(__name__)
RECURSIVE_ARCHITECTURES = {"tiny_recursive", "trm", "hrm", "transformer_act"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def cosine_schedule_with_warmup(
    step: int,
    *,
    base_lr: float,
    num_warmup_steps: int,
    num_training_steps: int,
    min_ratio: float = 0.0,
    num_cycles: float = 0.5,
) -> float:
    """Cosine decay learning rate schedule with linear warmup."""

    if step < num_warmup_steps:
        return base_lr * float(step + 1) / float(max(1, num_warmup_steps))

    if num_training_steps <= num_warmup_steps:
        return base_lr

    progress = float(step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    cycles = max(num_cycles, 1e-8)
    cosine_term = math.cos(math.pi * 2.0 * cycles * progress)
    scaled = min_ratio + max(0.0, (1.0 - min_ratio) * 0.5 * (1.0 + cosine_term))
    return base_lr * scaled


def forward_embedding_batch(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    *,
    is_recursive: bool,
) -> torch.Tensor:
    """Forward pass that optionally provides puzzle identifiers to the model."""

    inputs = batch["inputs"]
    targets = batch.get("targets")
    if is_recursive:
        puzzle_identifiers = batch["puzzle_identifiers"]
        return model(
            inputs,
            targets=targets,
            puzzle_identifiers=puzzle_identifiers,
        )

    if getattr(model, "requires_targets", False):
        return model(inputs, targets=targets)
    return model(inputs)


def evaluate_embedding(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    ignore_index: int,
    is_recursive: bool,
    tokens: TokenVocabulary,
    seed: int,
    tta_cfg: Optional[Dict[str, object]] = None,
    max_episodes: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate the model on the provided dataloader."""

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_episodes = 0
    support_loss = 0.0
    support_tokens = 0
    support_correct = 0
    support_episodes = 0
    query_loss = 0.0
    query_tokens = 0
    query_correct = 0
    query_episodes = 0
    tta_settings = dict(tta_cfg or {})
    tta_enabled = bool(tta_settings.get("enabled", False))
    num_augs = max(0, int(tta_settings.get("num_augs", 0)))
    vote_mode = str(tta_settings.get("vote", "mean_logits")).lower()
    apply_to_mode = str(tta_settings.get("apply_to", "query_only")).lower()
    palette_enabled = bool(tta_settings.get("use_palette", False))
    reverse_enabled = bool(tta_settings.get("use_reverse", False))

    if vote_mode not in {"mean_logits", "majority"}:
        raise ValueError(
            f"Unsupported TTA vote mode '{vote_mode}'. Expected 'mean_logits' or 'majority'."
        )
    if apply_to_mode not in {"query_only", "both"}:
        raise ValueError(
            f"Unsupported TTA apply_to='{apply_to_mode}'. Expected 'query_only' or 'both'."
        )

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch["labels"]
            sample_types = batch.get("sample_types")
            if sample_types is None:
                sample_types = torch.zeros(labels.size(0), dtype=torch.long, device=device)
            valid_mask = labels != ignore_index

            if not tta_enabled:
                logits_for_metrics = forward_embedding_batch(
                    model, batch, is_recursive=is_recursive
                )
                predictions_for_metrics = logits_for_metrics.argmax(dim=-1)
            else:
                base_logits = forward_embedding_batch(
                    model, batch, is_recursive=is_recursive
                )
                logits_sum = base_logits.clone()
                preds_views: List[torch.Tensor] = [base_logits.argmax(dim=-1)]
                num_views = 1

                aggregate_mask = valid_mask.clone()
                if apply_to_mode == "query_only":
                    query_mask = (sample_types == 1).unsqueeze(-1)
                    aggregate_mask = aggregate_mask & query_mask

                episode_start = total_episodes
                for aug_idx in range(num_augs):
                    rng_seed = seed + episode_start + aug_idx + batch_index
                    rng = random.Random(rng_seed)
                    perm = list(range(tokens.num_colors))
                    if palette_enabled and tokens.num_colors > 0:
                        perm = sample_palette_perm(rng, tokens.num_colors)
                    inv_perm = invert_perm(perm)

                    inputs_aug = batch["inputs"].clone()
                    targets_aug = batch["targets"].clone()
                    if palette_enabled and tokens.num_colors > 0:
                        inputs_aug = apply_palette(inputs_aug, perm, tokens.num_colors)
                        targets_aug = apply_palette(targets_aug, perm, tokens.num_colors)
                    if reverse_enabled:
                        inputs_aug = reverse_value_segments(inputs_aug, tokens)
                        targets_aug = reverse_value_segments(targets_aug, tokens)

                    aug_batch = dict(batch)
                    aug_batch["inputs"] = inputs_aug
                    aug_batch["targets"] = targets_aug
                    logits_aug = forward_embedding_batch(
                        model, aug_batch, is_recursive=is_recursive
                    )

                    if palette_enabled and tokens.num_colors > 0:
                        logits_aug = logits_aug.clone()
                        inv_indices = torch.tensor(
                            inv_perm,
                            dtype=torch.long,
                            device=logits_aug.device,
                        )
                        color_logits = logits_aug[..., : tokens.num_colors]
                        logits_aug[..., : tokens.num_colors] = color_logits.index_select(
                            -1, inv_indices
                        )

                    preds_aug = logits_aug.argmax(dim=-1)
                    if palette_enabled and tokens.num_colors > 0:
                        vocab_size = logits_aug.size(-1)
                        mapping = torch.arange(
                            vocab_size, device=logits_aug.device, dtype=torch.long
                        )
                        mapping[: tokens.num_colors] = torch.tensor(
                            inv_perm, device=logits_aug.device, dtype=torch.long
                        )
                        preds_aug = mapping[preds_aug]

                    logits_sum += logits_aug
                    num_views += 1
                    preds_views.append(preds_aug)

                mean_logits_all = logits_sum / float(num_views)
                mask_3d = aggregate_mask.unsqueeze(-1)
                final_logits = torch.where(mask_3d, mean_logits_all, base_logits)

                if aggregate_mask.any() and num_views > 1 and vote_mode == "majority":
                    stacked_preds = torch.stack(preds_views, dim=0)
                    stacked_preds_flat = stacked_preds.view(num_views, -1)
                    mask_flat = aggregate_mask.view(-1)
                    votes = stacked_preds_flat[:, mask_flat]
                    vocab_size = final_logits.size(-1)
                    if votes.numel() > 0:
                        vote_counts = torch.nn.functional.one_hot(
                            votes, num_classes=vocab_size
                        ).sum(dim=0)
                        winners = vote_counts.argmax(dim=-1)
                        max_counts = vote_counts.max(dim=-1, keepdim=True).values
                        tie_mask = (vote_counts == max_counts).sum(dim=-1) > 1
                        if tie_mask.any():
                            logits_flat = mean_logits_all.view(-1, vocab_size)[mask_flat]
                            tie_logits = logits_flat[tie_mask]
                            tie_choice = tie_logits.argmax(dim=-1)
                            winners = winners.clone()
                            winners[tie_mask] = tie_choice
                        final_preds_flat = preds_views[0].view(-1).clone()
                        final_preds_flat[mask_flat] = winners.to(final_preds_flat.dtype)
                        predictions_for_metrics = final_preds_flat.view_as(preds_views[0])
                    else:
                        predictions_for_metrics = preds_views[0]
                else:
                    aggregated_preds = mean_logits_all.argmax(dim=-1)
                    predictions_for_metrics = torch.where(
                        aggregate_mask, aggregated_preds, preds_views[0]
                    )

                logits_for_metrics = final_logits

            vocab_size = logits_for_metrics.size(-1)
            token_losses = torch.nn.functional.cross_entropy(
                logits_for_metrics.reshape(-1, vocab_size),
                labels.view(-1),
                ignore_index=ignore_index,
                reduction="none",
            ).view_as(labels)
            per_sample_loss = (token_losses * valid_mask).sum(dim=1)
            per_sample_tokens = valid_mask.long().sum(dim=1)
            per_sample_correct = (
                (predictions_for_metrics == labels) & valid_mask
            ).long().sum(dim=1)

            total_loss += float(per_sample_loss.sum().item())
            total_tokens += int(per_sample_tokens.sum().item())
            total_correct += int(per_sample_correct.sum().item())
            total_episodes += labels.size(0)

            support_mask = sample_types == 0
            query_mask = sample_types == 1
            if support_mask.any():
                support_loss += float(per_sample_loss[support_mask].sum().item())
                support_tokens += int(per_sample_tokens[support_mask].sum().item())
                support_correct += int(per_sample_correct[support_mask].sum().item())
                support_episodes += int(support_mask.sum().item())
            if query_mask.any():
                query_loss += float(per_sample_loss[query_mask].sum().item())
                query_tokens += int(per_sample_tokens[query_mask].sum().item())
                query_correct += int(per_sample_correct[query_mask].sum().item())
                query_episodes += int(query_mask.sum().item())

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
        **(
            {
                "support/loss": support_loss / support_tokens,
                "support/accuracy": support_correct / support_tokens
                if support_tokens
                else 0.0,
                "support/tokens": float(support_tokens),
                "support/episodes": float(support_episodes),
            }
            if support_tokens
            else {}
        ),
        **(
            {
                "query/loss": query_loss / query_tokens,
                "query/accuracy": query_correct / query_tokens
                if query_tokens
                else 0.0,
                "query/tokens": float(query_tokens),
                "query/episodes": float(query_episodes),
            }
            if query_tokens
            else {}
        ),
    }


def evaluate_embedding_splits(
    model: torch.nn.Module,
    loaders: Dict[str, DataLoader],
    *,
    device: torch.device,
    ignore_index: int,
    is_recursive: bool,
    tokens: TokenVocabulary,
    seed: int,
    tta_cfg: Optional[Dict[str, object]],
    max_episodes: Optional[int],
) -> Dict[str, Dict[str, float]]:
    """Evaluate the model across multiple splits."""

    return {
        split_name: evaluate_embedding(
            model,
            loader,
            device=device,
            ignore_index=ignore_index,
            is_recursive=is_recursive,
            tokens=tokens,
            seed=seed,
            tta_cfg=tta_cfg,
            max_episodes=max_episodes,
        )
        for split_name, loader in loaders.items()
    }


def scan_puzzle_statistics(
    datasets: Sequence[Tuple[str, Iterable[Dict[str, object]]]],
    *,
    identifier_table: PuzzleIdentifierTable,
) -> int:
    """Iterate over datasets once to populate puzzle ids and count samples."""

    total_pairs = 0
    for split_name, dataset in datasets:
        for episode in dataset:
            episode_id = str(episode.get("id"))
            identifier_table.get_or_create(f"{split_name}:{episode_id}")
            total_pairs += len(episode.get("train", []))
    return total_pairs


def prepare_embedding_model(
    cfg: DictConfig,
    tokens: TokenVocabulary,
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    num_puzzle_identifiers: int,
) -> torch.nn.Module:
    """Instantiate a baseline model with overrides suitable for embedding training."""

    architecture = str(cfg.model.architecture)
    registry = get_baseline_registry()
    if architecture not in registry:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Available baselines: {sorted(registry)}"
        )
    if architecture not in RECURSIVE_ARCHITECTURES:
        raise ValueError(
            "Puzzle-embedding training currently supports only recursive reasoning "
            f"architectures. Received '{architecture}'. Use training.mode=incontext "
            "for other baselines."
        )

    size_cfg = cfg.model.size
    model_kwargs: Dict[str, object] = {}
    if size_cfg and "variants" in size_cfg and architecture in size_cfg.variants:
        model_kwargs = OmegaConf.to_container(
            size_cfg.variants[architecture], resolve=True
        )  # type: ignore[assignment]

    model_kwargs = dict(model_kwargs or {})
    if architecture in RECURSIVE_ARCHITECTURES:
        model_kwargs["num_puzzle_identifiers"] = max(1, num_puzzle_identifiers)
        if "puzzle_emb_ndim" not in model_kwargs:
            model_kwargs["puzzle_emb_ndim"] = model_kwargs.get(
                "hidden_size", tokens.num_colors
            )

    baseline_config = BaselineConfig(
        input_vocab_size=tokens.vocab_size,
        output_vocab_size=tokens.vocab_size,
        max_seq_len=int(cfg.data.max_seq_len),
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        model_kwargs=model_kwargs,
    )
    model = create_baseline(architecture, baseline_config)
    return model.to(device=device, dtype=dtype)


def collect_puzzle_embeddings(model: torch.nn.Module) -> list[CastedSparseEmbedding]:
    """Return all sparse puzzle embedding modules registered on the model."""

    return [
        module
        for module in model.modules()
        if isinstance(module, CastedSparseEmbedding)
    ]


def make_training_dataloader(
    *,
    datasets: Sequence[Tuple[str, Iterable[Dict[str, object]]]],
    tokens: TokenVocabulary,
    cfg: DictConfig,
    identifier_table: PuzzleIdentifierTable,
    ignore_index: int,
    seed: int,
) -> DataLoader:
    """Construct the primary training dataloader for embedding supervision."""

    shuffle_enabled = bool(cfg.data.shuffle)
    if shuffle_enabled:
        buffer_cfg = cfg.data.get("shuffle_buffer", None)
        shuffle_buffer = int(buffer_cfg) if buffer_cfg is not None else max(
            int(cfg.data.batch_size) * 4, 256
        )
    else:
        shuffle_buffer = None

    iterable = PuzzleEmbeddingIterableDataset(
        datasets=datasets,
        tokens=tokens,
        ignore_index=ignore_index,
        max_seq_len=int(cfg.data.max_seq_len),
        drop_long=bool(cfg.data.drop_long_episodes),
        identifier_table=identifier_table,
        shuffle=shuffle_enabled,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )

    if int(cfg.data.num_workers) != 0:
        LOGGER.warning(
            "Puzzle embedding dataloader currently supports only num_workers=0. "
            "Overriding requested value %s.",
            cfg.data.num_workers,
        )

    collate_fn = build_puzzle_embedding_collate_fn(
        tokens,
        ignore_index=ignore_index,
        pad_to_length=int(cfg.data.max_seq_len),
    )

    return DataLoader(
        iterable,
        batch_size=int(cfg.data.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )


def make_evaluation_loader(
    *,
    split_name: str,
    dataset: Iterable[Dict[str, object]],
    tokens: TokenVocabulary,
    cfg: DictConfig,
    identifier_table: PuzzleIdentifierTable,
    batch_size: int,
    ignore_index: int,
) -> DataLoader:
    """Create a deterministic evaluation loader for a single split."""

    iterable = PuzzleEmbeddingIterableDataset(
        datasets=[(split_name, dataset)],
        tokens=tokens,
        ignore_index=ignore_index,
        max_seq_len=int(cfg.data.max_seq_len),
        drop_long=bool(cfg.data.drop_long_episodes),
        identifier_table=identifier_table,
        shuffle=False,
        shuffle_buffer=None,
        seed=int(cfg.seed),
        include_query=True,
    )

    collate_fn = build_puzzle_embedding_collate_fn(
        tokens,
        ignore_index=ignore_index,
        pad_to_length=int(cfg.data.max_seq_len),
    )

    return DataLoader(
        iterable,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )


def run(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    LOGGER.info("Working directory: %s", os.getcwd())
    LOGGER.info("Repository root: %s", REPO_ROOT)
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
    seed = int(cfg.seed)
    augmentation_settings = resolve_augmentation_settings(cfg, tokens)

    train_split = str(cfg.dataset.train_split)
    eval_split_names = normalize_split_config(cfg.dataset.get("eval_splits"))
    test_split_names = normalize_split_config(cfg.dataset.get("test_splits"))

    ordered_split_names: List[str] = []
    for split_name in [train_split, *eval_split_names, *test_split_names]:
        if split_name and split_name not in ordered_split_names:
            ordered_split_names.append(split_name)

    split_datasets: Dict[str, Iterator[Dict[str, object]]] = {}
    for split_name in ordered_split_names:
        split_datasets[split_name] = instantiate_episode_dataset(
            cfg,
            split_name,
            seed=seed,
            augmentation=augmentation_settings if split_name == train_split else None,
        )

    identifier_table = PuzzleIdentifierTable()
    total_pairs = scan_puzzle_statistics(
        [(name, split_datasets[name]) for name in ordered_split_names],
        identifier_table=identifier_table,
    )
    LOGGER.info(
        "Registered %d puzzle identifiers across %d samples.",
        identifier_table.size,
        total_pairs,
    )

    train_loader = make_training_dataloader(
        datasets=[(train_split, split_datasets[train_split])],
        tokens=tokens,
        cfg=cfg,
        identifier_table=identifier_table,
        ignore_index=ignore_index,
        seed=seed,
    )

    eval_batch_size = int(cfg.eval.batch_size)
    eval_tta_cfg = cfg.eval.get("tta", None)
    eval_loaders: Dict[str, DataLoader] = {}
    for split_name in eval_split_names:
        eval_loaders[split_name] = make_evaluation_loader(
            split_name=split_name,
            dataset=split_datasets[split_name],
            tokens=tokens,
            cfg=cfg,
            identifier_table=identifier_table,
            batch_size=eval_batch_size,
            ignore_index=ignore_index,
        )

    test_cfg = cfg.get("test", {})
    test_loaders: Dict[str, DataLoader] = {}
    if test_split_names:
        test_batch_size = int(test_cfg.get("batch_size", eval_batch_size))
        for split_name in test_split_names:
            test_loaders[split_name] = make_evaluation_loader(
                split_name=split_name,
                dataset=split_datasets[split_name],
                tokens=tokens,
                cfg=cfg,
                identifier_table=identifier_table,
                batch_size=test_batch_size,
                ignore_index=ignore_index,
            )
    max_eval_episodes = None
    eval_limit_cfg = cfg.eval.get("limit")
    if eval_limit_cfg is not None:
        max_eval_episodes = int(eval_limit_cfg)

    architecture = str(cfg.model.architecture)
    dtype_name = str(cfg.model.dtype)
    dtype = getattr(torch, dtype_name)
    requested_device = str(cfg.model.device)
    device = detect_device(requested_device)

    model = prepare_embedding_model(
        cfg,
        tokens,
        device=device,
        dtype=dtype,
        batch_size=int(cfg.data.batch_size),
        num_puzzle_identifiers=identifier_table.size,
    )
    is_recursive = architecture in RECURSIVE_ARCHITECTURES
    puzzle_embeddings: list[CastedSparseEmbedding] = []
    if is_recursive:
        puzzle_embeddings = collect_puzzle_embeddings(model)

    if wandb_module and wandb_run:
        num_parameters = sum(param.numel() for param in model.parameters())
        wandb_run.summary["model/num_parameters"] = int(num_parameters)

    recursive_cfg = cfg.get("recursive_trainer", {})
    puzzle_optimizer: Optional[PuzzleEmbeddingOptimizer] = None
    ema_helper: Optional[EMAHelper] = None
    if is_recursive:
        lr = float(recursive_cfg.get("lr", 1e-4))
        weight_decay = float(
            recursive_cfg.get("weight_decay", cfg.optimizer.weight_decay)
        )
        beta1 = float(recursive_cfg.get("beta1", 0.9))
        beta2 = float(recursive_cfg.get("beta2", 0.999))
        optimizer = AdamATan2(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
        warmup_steps = int(recursive_cfg.get("lr_warmup_steps", 2000))
        min_ratio = float(recursive_cfg.get("lr_min_ratio", 1.0))
        lr_cycles = float(recursive_cfg.get("lr_cycles", 0.5))
        puzzle_lr = float(recursive_cfg.get("puzzle_emb_lr", lr))
        puzzle_weight_decay = float(
            recursive_cfg.get("puzzle_emb_weight_decay", weight_decay)
        )
        if puzzle_embeddings and puzzle_lr != 0.0:
            puzzle_optimizer = PuzzleEmbeddingOptimizer(
                puzzle_embeddings,
                lr=puzzle_lr,
                weight_decay=puzzle_weight_decay,
            )
        scheduler = {
            "base_lr": lr,
            "warmup": warmup_steps,
            "min_ratio": min_ratio,
            "cycles": lr_cycles,
        }
        use_ema = bool(recursive_cfg.get("use_ema", False))
        if use_ema:
            ema_decay = float(recursive_cfg.get("ema_decay", 0.999))
            ema_helper = EMAHelper(decay=ema_decay)
            ema_helper.register(model)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.optimizer.lr),
            betas=tuple(cfg.optimizer.betas) if "betas" in cfg.optimizer else (0.9, 0.999),
            eps=float(cfg.optimizer.eps),
            weight_decay=float(cfg.optimizer.weight_decay),
        )
        warmup_steps = int(cfg.optimizer.warmup_steps)
        if warmup_steps > 0:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0),
            )
        else:
            scheduler = None
        puzzle_optimizer = None
        ema_helper = None

    recursive_grad_acc = recursive_cfg.get(
        "gradient_accumulation", cfg.trainer.gradient_accumulation
    )
    gradient_accumulation = max(1, int(recursive_grad_acc if is_recursive else cfg.trainer.gradient_accumulation))
    clip_grad_norm = cfg.trainer.clip_grad_norm

    micro_batches_per_epoch = math.ceil(
        total_pairs / max(1, int(cfg.data.batch_size))
    )
    updates_per_epoch = math.ceil(micro_batches_per_epoch / gradient_accumulation)

    num_epochs_cfg = cfg.trainer.get("num_epochs", None)
    num_epochs = int(num_epochs_cfg) if num_epochs_cfg is not None else None
    if num_epochs is not None and num_epochs <= 0:
        num_epochs = None

    max_steps_cfg = cfg.trainer.get("max_steps", None)
    max_steps = int(max_steps_cfg) if max_steps_cfg is not None else None
    if max_steps is not None and max_steps <= 0:
        max_steps = None

    if num_epochs is None and max_steps is None:
        raise ValueError(
            "Specify either trainer.num_epochs or trainer.max_steps for embedding training."
        )

    total_updates = max_steps if max_steps is not None else updates_per_epoch * int(num_epochs)

    log_every = int(cfg.trainer.log_every)
    eval_every_cfg = cfg.trainer.get("eval_every", None)
    eval_every = int(eval_every_cfg) if eval_every_cfg is not None else None
    max_eval_episodes_train = max_eval_episodes

    checkpoint_manager: Optional[CheckpointManager] = None
    checkpoint_cfg = cfg.trainer.get("checkpoints", None)
    if checkpoint_cfg and bool(checkpoint_cfg.get("enabled", False)):
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

    LOGGER.info("Starting embedding training on %s", device)
    global_step = 0
    current_epoch = 0
    last_train_metrics: Optional[Dict[str, float]] = None
    latest_eval_metrics: Optional[Dict[str, float]] = None
    latest_eval_by_split: Dict[str, Dict[str, float]] = {}
    last_step_stats: Optional[Dict[str, float]] = None

    while True:
        current_epoch += 1
        LOGGER.info("Epoch %d", current_epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        if puzzle_optimizer:
            puzzle_optimizer.zero_grad()

        step_loss = 0.0
        step_correct = 0
        step_tokens = 0
        micro_batches = 0

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}

            logits = forward_embedding_batch(
                model,
                batch,
                is_recursive=is_recursive,
            )
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

                if is_recursive:
                    lr_this_step = cosine_schedule_with_warmup(
                        step=global_step,
                        base_lr=scheduler["base_lr"],  # type: ignore[index]
                        num_warmup_steps=scheduler["warmup"],  # type: ignore[index]
                        num_training_steps=total_updates,
                        min_ratio=scheduler["min_ratio"],  # type: ignore[index]
                        num_cycles=scheduler["cycles"],  # type: ignore[index]
                    )
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_this_step

                optimizer.step()
                if puzzle_optimizer:
                    puzzle_optimizer.step()
                if not is_recursive and scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if puzzle_optimizer:
                    puzzle_optimizer.zero_grad()
                if ema_helper:
                    ema_helper.update(model)

                current_lr = optimizer.param_groups[0]["lr"]
                global_step += 1

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

                step_loss = 0.0
                step_correct = 0
                step_tokens = 0
                micro_batches = 0

                if max_steps is not None and global_step >= max_steps:
                    break

        if max_steps is not None and global_step >= max_steps:
            break

        if num_epochs is not None and current_epoch >= num_epochs:
            break

        if eval_every is not None and current_epoch % eval_every != 0:
            continue

        if eval_loaders:
            ema_backup = {}
            if ema_helper:
                ema_backup = ema_helper.swap_to_shadow(model)
            try:
                eval_metrics_by_split = evaluate_embedding_splits(
                    model,
                    eval_loaders,
                    device=device,
                    ignore_index=ignore_index,
                    is_recursive=is_recursive,
                    tokens=tokens,
                    seed=seed,
                    tta_cfg=eval_tta_cfg,
                    max_episodes=max_eval_episodes_train,
                )
            finally:
                if ema_helper:
                    ema_helper.restore(model, ema_backup)
            latest_eval_by_split = eval_metrics_by_split
            latest_eval_metrics = aggregate_split_metrics(eval_metrics_by_split)
            log_split_metrics(
                tag="val",
                metrics_by_split=eval_metrics_by_split,
                aggregated_metrics=latest_eval_metrics,
                step=global_step,
                epoch=current_epoch,
                wandb_module=wandb_module,
            )

            if checkpoint_manager:
                if ema_helper:
                    ema_backup = ema_helper.swap_to_shadow(model)
                else:
                    ema_backup = {}
                try:
                    checkpoint_manager.save_last(model)
                    checkpoint_manager.record_evaluation(
                        model=model,
                        aggregated_metrics=latest_eval_metrics or {},
                        metrics_by_split=latest_eval_by_split,
                        epoch=current_epoch,
                        step=global_step,
                    )
                finally:
                    if ema_helper:
                        ema_helper.restore(model, ema_backup)

        elif checkpoint_manager:
            if ema_helper:
                ema_backup = ema_helper.swap_to_shadow(model)
            else:
                ema_backup = {}
            try:
                checkpoint_manager.save_last(model)
            finally:
                if ema_helper:
                    ema_helper.restore(model, ema_backup)

    LOGGER.info("Finished training at step %d", global_step)

    test_metrics_by_split: Dict[str, Dict[str, float]] = {}
    test_aggregated_metrics: Dict[str, float] = {}
    if test_loaders:
        ema_backup = {}
        if ema_helper:
            ema_backup = ema_helper.swap_to_shadow(model)
        try:
            test_metrics_by_split = evaluate_embedding_splits(
                model,
                test_loaders,
                device=device,
                ignore_index=ignore_index,
                is_recursive=is_recursive,
                tokens=tokens,
                seed=seed,
                tta_cfg=eval_tta_cfg,
                max_episodes=test_cfg.get("limit"),
            )
        finally:
            if ema_helper:
                ema_helper.restore(model, ema_backup)
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
            if "support/loss" in latest_eval_metrics:
                wandb_run.summary["val/support_loss_mean"] = latest_eval_metrics.get(
                    "support/loss"
                )
                wandb_run.summary["val/support_accuracy_mean"] = latest_eval_metrics.get(
                    "support/accuracy"
                )
            if "query/loss" in latest_eval_metrics:
                wandb_run.summary["val/query_loss_mean"] = latest_eval_metrics.get(
                    "query/loss"
                )
                wandb_run.summary["val/query_accuracy_mean"] = latest_eval_metrics.get(
                    "query/accuracy"
                )
        for split_name, metrics in latest_eval_by_split.items():
            wandb_run.summary[f"val/{split_name}/loss"] = metrics.get("loss")
            wandb_run.summary[f"val/{split_name}/accuracy"] = metrics.get("accuracy")
            if "support/loss" in metrics:
                wandb_run.summary[f"val/{split_name}/support_loss"] = metrics.get(
                    "support/loss"
                )
                wandb_run.summary[f"val/{split_name}/support_accuracy"] = metrics.get(
                    "support/accuracy"
                )
            if "query/loss" in metrics:
                wandb_run.summary[f"val/{split_name}/query_loss"] = metrics.get(
                    "query/loss"
                )
                wandb_run.summary[f"val/{split_name}/query_accuracy"] = metrics.get(
                    "query/accuracy"
                )
        if test_aggregated_metrics:
            wandb_run.summary["test/loss_mean"] = test_aggregated_metrics.get("loss")
            wandb_run.summary["test/accuracy_mean"] = test_aggregated_metrics.get(
                "accuracy"
            )
            if "support/loss" in test_aggregated_metrics:
                wandb_run.summary["test/support_loss_mean"] = test_aggregated_metrics.get(
                    "support/loss"
                )
                wandb_run.summary[
                    "test/support_accuracy_mean"
                ] = test_aggregated_metrics.get("support/accuracy")
            if "query/loss" in test_aggregated_metrics:
                wandb_run.summary["test/query_loss_mean"] = test_aggregated_metrics.get(
                    "query/loss"
                )
                wandb_run.summary[
                    "test/query_accuracy_mean"
                ] = test_aggregated_metrics.get("query/accuracy")
        for split_name, metrics in test_metrics_by_split.items():
            wandb_run.summary[f"test/{split_name}/loss"] = metrics.get("loss")
            wandb_run.summary[f"test/{split_name}/accuracy"] = metrics.get("accuracy")
            if "support/loss" in metrics:
                wandb_run.summary[f"test/{split_name}/support_loss"] = metrics.get(
                    "support/loss"
                )
                wandb_run.summary[f"test/{split_name}/support_accuracy"] = metrics.get(
                    "support/accuracy"
                )
            if "query/loss" in metrics:
                wandb_run.summary[f"test/{split_name}/query_loss"] = metrics.get(
                    "query/loss"
                )
                wandb_run.summary[f"test/{split_name}/query_accuracy"] = metrics.get(
                    "query/accuracy"
                )
        wandb_run.finish()


@hydra.main(config_path="../configs", config_name="train/default", version_base=None)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
