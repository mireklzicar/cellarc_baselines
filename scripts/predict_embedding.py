#!/usr/bin/env python3
"""Generate predictions from a puzzle-embedding baseline on selected Cell ARC episodes."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_incontext import (  # noqa: E402
    TokenVocabulary,
    instantiate_episode_dataset,
    normalize_split_config,
    resolve_augmentation_settings,
    set_all_seeds,
    detect_device,
)
from scripts.train_embedding import (  # noqa: E402
    forward_embedding_batch,
    prepare_embedding_model,
    scan_puzzle_statistics,
)
from scripts.training.puzzle_embedding_dataset import (  # noqa: E402
    PuzzleEmbeddingSample,
    PuzzleIdentifierTable,
    build_puzzle_embedding_collate_fn,
    build_query_sample,
)


LOGGER = logging.getLogger("predict_embedding")


@dataclass(frozen=True)
class EpisodeSpec:
    """Episode metadata parsed from the reference JSONL prediction log."""

    split_label: str
    base_split: str
    episode_id: str
    episode_index: int
    position: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate transformer embedding predictions for specific Cell ARC episodes.",
    )
    default_dataset_root = REPO_ROOT / "outputs" / "hf_datasets"
    default_output = REPO_ROOT / "outputs" / "neural" / "transformer_large_embedding_test100_predictions.jsonl"
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to the model_best.pt checkpoint.")
    parser.add_argument("--reference-log", required=True, type=Path, help="JSONL log containing the target episodes.")
    parser.add_argument("--output", type=Path, default=default_output, help="Where to write the prediction JSONL log.")
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root, help="Local Hugging Face cache root.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--device", default="auto", help="Computation device (auto|cpu|cuda|mps|cuda:0, ...).")
    parser.add_argument("--model-name", default="transformer_large_embedding", help="Name recorded in the log output.")
    parser.add_argument("--model-size", default="large", help="Hydra size preset to load (small|medium|large).")
    parser.add_argument("--model-architecture", default="transformer", help="Baseline architecture to instantiate.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed used when instantiating datasets.")
    parser.add_argument("--include-metadata", action="store_true", help="Whether to download companion metadata repos.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def _normalize_reference_split(name: str) -> str:
    suffixes = ("_100",)
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_reference_specs(path: Path) -> Dict[str, List[EpisodeSpec]]:
    specs: Dict[str, List[EpisodeSpec]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            split_label = str(data.get("split") or "")
            if not split_label:
                raise ValueError(f"Missing 'split' key in reference record at line {position}.")
            base_split = _normalize_reference_split(split_label)
            episode_id = str(data.get("episode_id") or "")
            if not episode_id:
                raise ValueError(f"Missing 'episode_id' in reference record at line {position}.")
            episode_index = int(data.get("episode_index", position))
            spec = EpisodeSpec(
                split_label=split_label,
                base_split=base_split,
                episode_id=episode_id,
                episode_index=episode_index,
                position=position,
            )
            specs.setdefault(base_split, []).append(spec)
    if not specs:
        raise ValueError(f"No episodes were parsed from '{path}'.")
    return specs


def load_config(args: argparse.Namespace) -> DictConfig:
    overrides = [
        f"model/size={args.model_size}",
        f"model.architecture={args.model_architecture}",
        "training.mode=embedding",
        f"dataset.root={args.dataset_root.expanduser().resolve()}",
        f"dataset.include_metadata={'true' if args.include_metadata else 'false'}",
        f"seed={args.seed}",
        f"data.batch_size={args.batch_size}",
        f"eval.batch_size={args.batch_size}",
        f"model.device={args.device}",
    ]
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train/default", overrides=overrides)
    return cfg


def build_identifier_table(
    cfg: DictConfig,
    tokens: TokenVocabulary,
    *,
    seed: int,
) -> Tuple[PuzzleIdentifierTable, Dict[str, Iterable[Dict[str, Any]]]]:
    train_split = str(cfg.dataset.train_split)
    eval_split_names = normalize_split_config(cfg.dataset.get("eval_splits"))
    test_split_names = normalize_split_config(cfg.dataset.get("test_splits"))
    ordered_split_names: List[str] = []
    for split_name in [train_split, *eval_split_names, *test_split_names]:
        if split_name and split_name not in ordered_split_names:
            ordered_split_names.append(split_name)

    augmentation_settings = resolve_augmentation_settings(cfg, tokens)

    split_datasets: Dict[str, Iterable[Dict[str, Any]]] = {}
    for split_name in ordered_split_names:
        split_datasets[split_name] = instantiate_episode_dataset(
            cfg,
            split_name,
            seed=seed,
            augmentation=augmentation_settings if split_name == train_split else None,
        )

    identifier_table = PuzzleIdentifierTable()
    scan_puzzle_statistics(
        [(name, split_datasets[name]) for name in ordered_split_names],
        identifier_table=identifier_table,
    )
    LOGGER.info("Registered %d puzzle identifiers.", identifier_table.size)
    return identifier_table, split_datasets


def collect_query_samples(
    *,
    cfg: DictConfig,
    split_name: str,
    dataset: Iterable[Mapping[str, Any]],
    specs: Sequence[EpisodeSpec],
    identifier_table: PuzzleIdentifierTable,
    tokens: TokenVocabulary,
    ignore_index: int,
) -> Tuple[List[PuzzleEmbeddingSample], List[Dict[str, Any]]]:
    needed_ids = {spec.episode_id for spec in specs}
    by_id: Dict[str, Mapping[str, Any]] = {}
    for episode in dataset:
        episode_id = str(episode.get("id"))
        if episode_id in needed_ids and episode_id not in by_id:
            by_id[episode_id] = episode
            if len(by_id) == len(needed_ids):
                break
    missing = sorted(episode_id for episode_id in needed_ids if episode_id not in by_id)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} episodes on split '{split_name}': {missing[:5]}")

    samples: List[PuzzleEmbeddingSample] = []
    metadata: List[Dict[str, Any]] = []
    max_seq_len = int(cfg.data.max_seq_len)
    drop_long = bool(cfg.data.drop_long_episodes)

    for spec in specs:
        episode = by_id[spec.episode_id]
        puzzle_key = f"{split_name}:{spec.episode_id}"
        puzzle_identifier = identifier_table.get(puzzle_key)
        if puzzle_identifier is None:
            raise RuntimeError(f"Puzzle identifier for key '{puzzle_key}' was not registered.")
        sample = build_query_sample(
            puzzle_identifier=puzzle_identifier,
            episode=episode,
            tokens=tokens,
            ignore_index=ignore_index,
            max_seq_len=max_seq_len,
            drop_long=drop_long,
        )
        if sample is None:
            raise RuntimeError(f"Episode {spec.episode_id} exceeded max_seq_len={max_seq_len}.")
        samples.append(sample)
        metadata.append(
            {
                "spec": spec,
                "episode": episode,
                "puzzle_identifier": puzzle_identifier,
            }
        )

    return samples, metadata


def predict_split(
    *,
    model: torch.nn.Module,
    samples: Sequence[PuzzleEmbeddingSample],
    metadata: Sequence[Dict[str, Any]],
    tokens: TokenVocabulary,
    ignore_index: int,
    batch_size: int,
    device: torch.device,
    desc: str,
    max_seq_len: int,
    model_name: str,
    checkpoint: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    if len(samples) != len(metadata):
        raise ValueError("Sample and metadata lengths differ.")
    collate_fn = build_puzzle_embedding_collate_fn(
        tokens,
        ignore_index=ignore_index,
        pad_to_length=max_seq_len,
    )
    loader = DataLoader(
        list(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    progress = tqdm(total=len(samples), desc=desc)
    model.eval()
    records: List[Dict[str, Any]] = []
    cursor = 0
    total_tokens = 0
    total_correct = 0
    exact_matches = 0

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = forward_embedding_batch(
                model,
                batch,
                use_puzzle_identifiers=True,
            )
            predictions = logits.argmax(dim=-1)
            labels = batch["labels"]
            valid_mask = labels != ignore_index
            loss_mask = batch["loss_mask"].bool()

            total_tokens += int(valid_mask.sum().item())
            total_correct += int(((predictions == labels) & valid_mask).sum().item())

            batch_size_effective = predictions.size(0)
            for row in range(batch_size_effective):
                meta = metadata[cursor]
                cursor += 1
                spec: EpisodeSpec = meta["spec"]
                episode = meta["episode"]
                solution_mask = loss_mask[row]
                pred_solution_tensor = predictions[row][solution_mask].detach().cpu()
                target_solution_tensor = labels[row][solution_mask].detach().cpu()
                pred_solution = pred_solution_tensor.tolist()
                target_solution = target_solution_tensor.tolist()
                pred_solution = [int(value) for value in pred_solution]
                target_solution = [int(value) for value in target_solution]
                is_exact = pred_solution == target_solution
                exact_matches += int(is_exact)

                records.append(
                    {
                        "model": model_name,
                        "checkpoint": str(checkpoint),
                        "split": spec.split_label,
                        "base_split": spec.base_split,
                        "episode_id": spec.episode_id,
                        "episode_index": spec.episode_index,
                        "position": spec.position,
                        "puzzle_identifier": int(meta["puzzle_identifier"]),
                        "prediction": pred_solution,
                        "target": target_solution,
                        "query": [int(v) for v in (episode.get("query") or [])],
                        "solution": [int(v) for v in (episode.get("solution") or [])],
                        "is_exact_match": is_exact,
                    }
                )
            progress.update(batch_size_effective)

    progress.close()
    token_accuracy = total_correct / total_tokens if total_tokens else 0.0
    episode_accuracy = exact_matches / len(samples) if samples else 0.0
    metrics = {
        "token_accuracy": token_accuracy,
        "episode_accuracy": episode_accuracy,
        "episodes": len(samples),
        "tokens": total_tokens,
    }
    return records, metrics


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    reference_specs = load_reference_specs(args.reference_log)
    LOGGER.info(
        "Loaded %d target splits from %s.",
        len(reference_specs),
        args.reference_log,
    )

    cfg = load_config(args)
    dataset_root = Path(cfg.dataset.root).expanduser().resolve()
    LOGGER.info("Dataset root: %s", dataset_root)
    LOGGER.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg))

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

    identifier_table, split_datasets = build_identifier_table(
        cfg,
        tokens,
        seed=int(cfg.seed),
    )

    device = detect_device(str(cfg.model.device))
    dtype = getattr(torch, str(cfg.model.dtype))
    model = prepare_embedding_model(
        cfg,
        tokens,
        device=device,
        dtype=dtype,
        batch_size=int(cfg.data.batch_size),
        num_puzzle_identifiers=identifier_table.size,
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    LOGGER.info("Loaded checkpoint %s onto %s.", args.checkpoint, device)

    all_records: List[Dict[str, Any]] = []
    per_split_metrics: Dict[str, Dict[str, float]] = {}
    for base_split, specs in reference_specs.items():
        dataset = split_datasets.get(base_split)
        if dataset is None:
            raise RuntimeError(
                f"Reference split '{base_split}' was not part of the training configuration."
            )
        samples, metadata = collect_query_samples(
            cfg=cfg,
            split_name=base_split,
            dataset=dataset,
            specs=specs,
            identifier_table=identifier_table,
            tokens=tokens,
            ignore_index=ignore_index,
        )
        if not samples:
            LOGGER.warning("No samples collected for split %s; skipping.", base_split)
            continue
        records, metrics = predict_split(
            model=model,
            samples=samples,
            metadata=metadata,
            tokens=tokens,
            ignore_index=ignore_index,
            batch_size=args.batch_size,
            device=device,
            desc=f"{base_split} predictions",
            max_seq_len=int(cfg.data.max_seq_len),
            model_name=args.model_name,
            checkpoint=args.checkpoint,
        )
        per_split_metrics[base_split] = metrics
        all_records.extend(records)
        LOGGER.info(
            "[%s] token_acc=%.4f episode_acc=%.4f (%d episodes)",
            base_split,
            metrics["token_accuracy"],
            metrics["episode_accuracy"],
            metrics["episodes"],
        )

    if not all_records:
        raise RuntimeError("No predictions were generated.")

    all_records.sort(key=lambda record: record["position"])
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in all_records:
            json.dump(record, handle)
            handle.write("\n")

    LOGGER.info("Wrote %d predictions to %s.", len(all_records), output_path)
    for split_name, metrics in per_split_metrics.items():
        LOGGER.info(
            "Final %s metrics: token_acc=%.4f episode_acc=%.4f",
            split_name,
            metrics["token_accuracy"],
            metrics["episode_accuracy"],
        )


if __name__ == "__main__":
    main()
