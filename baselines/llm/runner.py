"""Utilities for evaluating LLM baselines over benchmark splits."""

from __future__ import annotations

import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm.auto import tqdm

from cellarc import EpisodeDataset, download_benchmark

from .client import OpenAIClient, OpenAIClientConfig
from .prompting import PromptBuilder, PromptBuilderConfig
from .solver import LLMPrediction, LLMSolver, PredictionParser, PredictionParserConfig

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

    def update(self, prediction: List[int], target: List[int]) -> None:
        self.episodes += 1
        if prediction == target:
            self.solved += 1

        for pred_symbol, target_symbol in zip_longest(
            prediction,
            target,
            fillvalue=None,
        ):
            self.token_total += 1
            if pred_symbol == target_symbol:
                self.token_correct += 1


class LLMEvaluator:
    """Coordinates dataset loading, prompting, and scoring."""

    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        self._prediction_log_path: Optional[Path] = None
        log_path = getattr(cfg.eval, "prediction_log", None)
        if log_path:
            resolved = Path(to_absolute_path(log_path))
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self._prediction_log_path = resolved

        instructions_text = _read_prompt_instructions(cfg.prompt.instructions_path)

        self._prompt_builder = PromptBuilder(
            PromptBuilderConfig(
                instructions=instructions_text,
                system_message=cfg.prompt.system_message,
                response_format_hint=cfg.prompt.response_format_hint,
                max_train_examples=cfg.prompt.max_train_examples,
                include_episode_id=cfg.prompt.include_episode_id,
            )
        )
        self._parser = PredictionParser(
            PredictionParserConfig(
                clip_to_expected=cfg.parser.clip_to_expected,
                allow_partial=cfg.parser.allow_partial,
            )
        )
        self._base_llm_config = OpenAIClientConfig(
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
            top_p=cfg.llm.top_p,
            max_output_tokens=cfg.llm.max_output_tokens,
            reasoning_effort=cfg.llm.reasoning_effort,
            api_style=cfg.llm.api_style,
            api_key_env=cfg.llm.api_key_env,
            base_url=cfg.llm.base_url,
            organization=cfg.llm.organization,
            request_timeout=cfg.llm.request_timeout,
            max_retries=cfg.llm.max_retries,
            retry_backoff=cfg.llm.retry_backoff,
            stream=getattr(cfg.llm, "stream", False),
            use_async=getattr(cfg.llm, "use_async", False),
        )

    def evaluate_split(self, split: str) -> SplitMetrics:
        dataset = _load_dataset_with_fallback(self._cfg, split)
        progress = self._cfg.eval.progress_bar
        max_episodes = self._cfg.eval.max_episodes
        log_every = self._cfg.eval.log_every_n

        base_iter: Iterable[Mapping[str, Any]]
        total_hint = _maybe_len(dataset)
        if progress:
            base_iter = tqdm(dataset, total=total_hint, desc=f"{split} episodes")
        else:
            base_iter = dataset

        pending: List[tuple[int, Mapping[str, Any], List[int]]] = []
        for index, episode in enumerate(base_iter):
            if max_episodes is not None and index >= max_episodes:
                break

            solution = episode.get("solution")
            if solution is None:
                LOGGER.warning("Skipping episode %s without a ground-truth solution.", index)
                continue

            solution_list = list(solution)
            pending.append((index, episode, solution_list))

        if not pending:
            return SplitMetrics(split=split)

        if getattr(self._cfg.llm, "use_async", False):
            return asyncio.run(self._evaluate_split_async(split, pending, log_every))

        return self._evaluate_split_threaded(split, pending, log_every)

    def _evaluate_split_threaded(
        self,
        split: str,
        pending: List[Tuple[int, Mapping[str, Any], List[int]]],
        log_every: Optional[int],
    ) -> SplitMetrics:
        worker_cfg = getattr(self._cfg.eval, "num_workers", 1) or 1
        max_workers = max(1, min(int(worker_cfg), len(pending)))

        metrics = SplitMetrics(split=split)
        processed = 0
        progress_bar = (
            tqdm(total=len(pending), desc=f"{split} predictions")
            if getattr(self._cfg.eval, "progress_bar", False)
            else None
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self._predict_episode, episode, solution_list): (
                    episode_index,
                    episode,
                    solution_list,
                )
                for episode_index, episode, solution_list in pending
            }

            for future in as_completed(future_map):
                episode_index, episode, solution_list = future_map[future]
                try:
                    prediction = future.result()
                except Exception as exc:  # pragma: no cover - logging path
                    LOGGER.error(
                        "Episode %s on split %s failed: %s", episode_index, split, exc
                    )
                    continue

                metrics.update(prediction.tokens, solution_list)
                self._log_prediction(split, episode_index, episode, prediction, solution_list)

                processed += 1
                if progress_bar:
                    progress_bar.update(1)
                if log_every and processed % log_every == 0:
                    LOGGER.info(
                        "[%s] processed %s episodes — solved %s (%s) / token accuracy %s",
                        split,
                        metrics.episodes,
                        metrics.solved,
                        _format_percentage(metrics.episode_accuracy),
                        _format_percentage(metrics.token_accuracy),
                    )

        if progress_bar:
            progress_bar.close()
        return metrics

    async def _evaluate_split_async(
        self,
        split: str,
        pending: List[Tuple[int, Mapping[str, Any], List[int]]],
        log_every: Optional[int],
    ) -> SplitMetrics:
        concurrency_cfg = getattr(self._cfg.eval, "async_concurrency", len(pending)) or len(pending)
        max_concurrency = max(1, min(int(concurrency_cfg), len(pending)))
        semaphore = asyncio.Semaphore(max_concurrency)

        async def run_episode(item: Tuple[int, Mapping[str, Any], List[int]]):
            episode_index, episode, solution_list = item
            solver = self._make_solver(async_mode=True)
            expected_length = len(solution_list)
            async with semaphore:
                prediction = await solver.predict_async(episode, expected_length)
            return episode_index, episode, solution_list, prediction

        tasks = [asyncio.create_task(run_episode(item)) for item in pending]
        metrics = SplitMetrics(split=split)
        processed = 0
        progress_bar = (
            tqdm(total=len(pending), desc=f"{split} predictions")
            if getattr(self._cfg.eval, "progress_bar", False)
            else None
        )

        for task in asyncio.as_completed(tasks):
            try:
                episode_index, episode, solution_list, prediction = await task
            except Exception as exc:  # pragma: no cover - logging path
                LOGGER.error("Async episode failed on split %s: %s", split, exc)
                continue

            metrics.update(prediction.tokens, solution_list)
            self._log_prediction(split, episode_index, episode, prediction, solution_list)

            processed += 1
            if progress_bar:
                progress_bar.update(1)
            if log_every and processed % log_every == 0:
                LOGGER.info(
                    "[%s] processed %s episodes — solved %s (%s) / token accuracy %s",
                    split,
                    metrics.episodes,
                    metrics.solved,
                    _format_percentage(metrics.episode_accuracy),
                    _format_percentage(metrics.token_accuracy),
                    )

        if progress_bar:
            progress_bar.close()
        return metrics

    def _log_prediction(
        self,
        split: str,
        episode_index: int,
        episode: Mapping[str, Any],
        prediction: LLMPrediction,
        solution: Sequence[int],
    ) -> None:
        if not self._prediction_log_path:
            return

        query = episode.get("query")
        record = {
            "split": split,
            "episode_index": episode_index,
            "episode_id": episode.get("id"),
            "raw_response": prediction.raw_text,
            "parsed_output": prediction.tokens,
            "prompt": prediction.prompt_text,
            "query": list(query) if isinstance(query, list) else query,
            "solution": list(solution),
        }

        with self._prediction_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record))
            handle.write("\n")

    def _predict_episode(
        self,
        episode: Mapping[str, Any],
        solution: Sequence[int],
    ) -> LLMPrediction:
        solver = self._make_solver(async_mode=False)
        expected_length = len(solution)
        return solver.predict(episode, expected_length)

    def _make_solver(self, async_mode: bool = False) -> LLMSolver:
        config = self._base_llm_config
        if config.use_async != async_mode:
            config = replace(config, use_async=async_mode)
        client = OpenAIClient(config)
        return LLMSolver(client, self._prompt_builder, self._parser)


def evaluate_splits(cfg: DictConfig) -> List[SplitMetrics]:
    evaluator = LLMEvaluator(cfg)
    results: List[SplitMetrics] = []
    for split in cfg.splits:
        LOGGER.info("Evaluating split %s", split)
        metrics = evaluator.evaluate_split(split)
        results.append(metrics)
        LOGGER.info(
            "Split %s — solved %s/%s (%s), token accuracy %s",
            split,
            metrics.solved,
            metrics.episodes,
            _format_percentage(metrics.episode_accuracy),
            _format_percentage(metrics.token_accuracy),
        )
    return results


def _read_prompt_instructions(path_str: str) -> str:
    if not path_str:
        raise ValueError("prompt.instructions_path must be provided.")

    absolute_path = Path(to_absolute_path(path_str))
    if not absolute_path.exists():
        raise FileNotFoundError(f"Prompt instructions file not found: {absolute_path}")
    return absolute_path.read_text().strip()


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
    except (KeyError, FileNotFoundError):
        LOGGER.info("Falling back to local dataset look-up for split %s", split)

    repository_path = download_benchmark(
        name=cfg.dataset.benchmark,
        include_metadata=cfg.dataset.include_metadata,
        root=dataset_root,
    )

    split_key = _normalize_split_name(split)
    data_dir = repository_path / "data"

    manifest_path = data_dir / split_key / f"{split_key}_manifest.json"
    if manifest_path.exists():
        return EpisodeDataset(manifest=manifest_path, **dataset_kwargs)

    json_candidate = data_dir / f"{split_key}.jsonl"
    parquet_candidate = data_dir / f"{split_key}.parquet"

    if json_candidate.exists():
        return EpisodeDataset(paths=[json_candidate], **dataset_kwargs)

    if parquet_candidate.exists():
        return EpisodeDataset(paths=[parquet_candidate], **dataset_kwargs)

    raise FileNotFoundError(
        f"Could not locate data for split '{split}' in repository '{repository_path}'."
    )


def _format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"
