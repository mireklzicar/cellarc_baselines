#!/usr/bin/env python3
"""Evaluate GPT-5 (or similar) LLM baselines on Cellular ARC splits."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Also look for a sibling checkout of the cellarc package if it is not installed.
CELLARC_REPO = REPO_ROOT.parent / "cellarc"
if CELLARC_REPO.exists() and str(CELLARC_REPO) not in sys.path:
    sys.path.insert(0, str(CELLARC_REPO))

from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig

from baselines.llm.runner import SplitMetrics, evaluate_splits

LOGGER = logging.getLogger(__name__)

load_dotenv()


@hydra.main(config_path="../configs/llm", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=getattr(logging, str(cfg.logging.level).upper(), logging.INFO))
    LOGGER.info(
        "Evaluating LLM model %s on splits: %s",
        cfg.llm.model,
        ", ".join(cfg.splits),
    )
    results = evaluate_splits(cfg)
    _log_summary(results)


def _log_summary(results: List[SplitMetrics]) -> None:
    total_episodes = sum(m.episodes for m in results)
    total_solved = sum(m.solved for m in results)
    total_tokens = sum(m.token_total for m in results)
    total_correct_tokens = sum(m.token_correct for m in results)

    def fmt(value: float) -> str:
        return f"{value * 100:.2f}%"

    overall_episode_accuracy = (total_solved / total_episodes) if total_episodes else 0.0
    overall_token_accuracy = (total_correct_tokens / total_tokens) if total_tokens else 0.0

    LOGGER.info("=== Summary ===")
    for metrics in results:
        LOGGER.info(
            "%s: solved %s/%s (%s), token accuracy %s",
            metrics.split,
            metrics.solved,
            metrics.episodes,
            fmt(metrics.episode_accuracy),
            fmt(metrics.token_accuracy),
        )

    LOGGER.info(
        "Overall: solved %s/%s episodes (%s), token accuracy %s",
        total_solved,
        total_episodes,
        fmt(overall_episode_accuracy),
        fmt(overall_token_accuracy),
    )


if __name__ == "__main__":
    main()
