#!/usr/bin/env python3
"""Evaluate symbolic baselines on Cell ARC benchmark splits."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Also look for a sibling checkout of the cellarc package if it is not installed.
CELLARC_REPO = REPO_ROOT.parent / "cellarc"
if CELLARC_REPO.exists() and str(CELLARC_REPO) not in sys.path:
    sys.path.insert(0, str(CELLARC_REPO))

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from baselines.symbolic import SymbolicSolver, create_symbolic_baseline
from baselines.symbolic.utils import clone_as_ints, iter_symbols
from cellarc import EpisodeDataset, download_benchmark

LOGGER = logging.getLogger(__name__)


@dataclass
class SplitMetrics:
    split: str
    episodes: int = 0
    solved: int = 0
    token_total: int = 0
    token_correct: int = 0

    @property
    def episode_accuracy(self) -> float:
        return (self.solved / self.episodes) if self.episodes else 0.0

    @property
    def token_accuracy(self) -> float:
        return (self.token_correct / self.token_total) if self.token_total else 0.0


def _maybe_len(obj: Iterable[Any]) -> Optional[int]:
    try:
        return len(obj)  # type: ignore[arg-type]
    except Exception:
        return None


def _normalize_split_name(split: str) -> str:
    parts = split.lower().replace("-", "_").replace("/", "_").split()
    return "_".join(part for part in parts if part)


def _load_dataset_with_fallback(cfg: DictConfig, split: str) -> EpisodeDataset:
    dataset_root = cfg.dataset.root
    if dataset_root is not None:
        dataset_root = to_absolute_path(dataset_root)

    dataset_kwargs: dict[str, Any] = {}
    try:
        return EpisodeDataset.from_huggingface(
            split=split,
            name=cfg.dataset.benchmark,
            include_metadata=cfg.dataset.include_metadata,
            root=dataset_root,
            **dataset_kwargs,
        )
    except (KeyError, FileNotFoundError) as exc:
        LOGGER.info(
            "Falling back to direct file lookup for split %s after error: %s", split, exc
        )

    repository_path = download_benchmark(
        name=cfg.dataset.benchmark,
        include_metadata=cfg.dataset.include_metadata,
        root=dataset_root,
    )

    split_key = _normalize_split_name(split)
    data_dir = repository_path / "data"

    manifest_path = data_dir / split_key / f"{split_key}_manifest.json"
    if manifest_path.exists():
        LOGGER.debug("Using manifest %s for split %s", manifest_path, split)
        return EpisodeDataset(manifest=manifest_path, **dataset_kwargs)

    json_candidate = data_dir / f"{split_key}.jsonl"
    parquet_candidate = data_dir / f"{split_key}.parquet"

    if json_candidate.exists():
        LOGGER.debug("Using JSONL file %s for split %s", json_candidate, split)
        return EpisodeDataset(paths=[json_candidate], **dataset_kwargs)

    if parquet_candidate.exists():
        LOGGER.debug("Using Parquet file %s for split %s", parquet_candidate, split)
        return EpisodeDataset(paths=[parquet_candidate], **dataset_kwargs)

    raise FileNotFoundError(
        f"Could not locate data for split '{split}' in repository '{repository_path}'."
    )


def evaluate_split(cfg: DictConfig, split: str, solver: SymbolicSolver) -> SplitMetrics:
    dataset = _load_dataset_with_fallback(cfg, split)

    max_episodes = cfg.eval.max_episodes
    progress = cfg.eval.progress_bar
    total_hint = _maybe_len(dataset)

    iterator: Iterable[Any]
    if progress:
        iterator = tqdm(dataset, total=total_hint, desc=f"{split} episodes")
    else:
        iterator = dataset

    metrics = SplitMetrics(split=split)

    for index, episode in enumerate(iterator):
        if max_episodes is not None and index >= max_episodes:
            break

        solution = episode.get("solution")
        if solution is None:
            LOGGER.warning("Skipping episode %s without ground-truth solution.", index)
            continue

        prediction_raw = solver.predict(episode)
        prediction = clone_as_ints(prediction_raw)
        target = clone_as_ints(solution)

        metrics.episodes += 1
        if prediction == target:
            metrics.solved += 1

        pred_tokens = list(iter_symbols(prediction))
        target_tokens = list(iter_symbols(target))
        for pred_symbol, target_symbol in zip_longest(pred_tokens, target_tokens, fillvalue=None):
            metrics.token_total += 1
            if pred_symbol == target_symbol:
                metrics.token_correct += 1

    return metrics


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def _serialize_metrics(metrics: SplitMetrics) -> dict[str, Any]:
    return {
        "split": metrics.split,
        "episodes": metrics.episodes,
        "solved": metrics.solved,
        "episode_accuracy": metrics.episode_accuracy,
        "token_total": metrics.token_total,
        "token_correct": metrics.token_correct,
        "token_accuracy": metrics.token_accuracy,
    }


def _aggregate_overall(results: List[SplitMetrics]) -> dict[str, float]:
    total_episodes = sum(m.episodes for m in results)
    total_solved = sum(m.solved for m in results)
    total_tokens = sum(m.token_total for m in results)
    total_correct_tokens = sum(m.token_correct for m in results)

    episode_accuracy = (total_solved / total_episodes) if total_episodes else 0.0
    token_accuracy = (total_correct_tokens / total_tokens) if total_tokens else 0.0

    return {
        "episodes": total_episodes,
        "solved": total_solved,
        "episode_accuracy": episode_accuracy,
        "token_total": total_tokens,
        "token_correct": total_correct_tokens,
        "token_accuracy": token_accuracy,
    }


def _resolve_results_path(cfg: DictConfig) -> Path:
    configured = getattr(cfg.eval, "results_json_path", None)
    if configured:
        path = Path(to_absolute_path(str(configured)))
    else:
        default_dir = REPO_ROOT / "outputs" / "symbolic"
        path = default_dir / f"{cfg.baseline.name}_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _build_results_payload(
    cfg: DictConfig,
    results: List[SplitMetrics],
    overall: dict[str, float],
) -> dict[str, Any]:
    baseline_section: dict[str, Any] = {"name": cfg.baseline.name}
    if "kwargs" in cfg.baseline:
        baseline_kwargs = OmegaConf.to_container(cfg.baseline.kwargs, resolve=True)
        if baseline_kwargs:
            baseline_section["kwargs"] = baseline_kwargs

    payload = {
        "baseline": baseline_section,
        "benchmark": cfg.dataset.benchmark,
        "splits": [_serialize_metrics(m) for m in results],
        "overall": overall,
        "max_episodes": cfg.eval.max_episodes,
    }
    return payload


def _write_results_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


@hydra.main(config_path="../configs/symbolic", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=getattr(logging, str(cfg.logging.level).upper(), logging.INFO))
    LOGGER.info(
        "Evaluating symbolic baseline %s on splits: %s",
        cfg.baseline.name,
        ", ".join(cfg.splits),
    )

    results: List[SplitMetrics] = []
    for split in cfg.splits:
        LOGGER.info("Starting split %s", split)
        solver_kwargs = (
            OmegaConf.to_container(cfg.baseline.kwargs, resolve=True)
            if "kwargs" in cfg.baseline
            else {}
        )
        solver = create_symbolic_baseline(cfg.baseline.name, **(solver_kwargs or {}))
        metrics = evaluate_split(cfg, split, solver)
        results.append(metrics)
        LOGGER.info(
            "Split %s — solved %s/%s episodes (%s), token accuracy %s",
            split,
            metrics.solved,
            metrics.episodes,
            format_percentage(metrics.episode_accuracy),
            format_percentage(metrics.token_accuracy),
        )

    overall_summary = _aggregate_overall(results)

    LOGGER.info("=== Summary ===")
    for metrics in results:
        LOGGER.info(
            "%s: solved %s/%s (%s), token accuracy %s",
            metrics.split,
            metrics.solved,
            metrics.episodes,
            format_percentage(metrics.episode_accuracy),
            format_percentage(metrics.token_accuracy),
        )

    LOGGER.info(
        "Overall: solved %s/%s episodes (%s), token accuracy %s",
        overall_summary["solved"],
        overall_summary["episodes"],
        format_percentage(overall_summary["episode_accuracy"]),
        format_percentage(overall_summary["token_accuracy"]),
    )

    results_path = _resolve_results_path(cfg)
    payload = _build_results_payload(cfg, results, overall_summary)
    _write_results_json(results_path, payload)
    LOGGER.info("Wrote metrics JSON to %s", results_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
