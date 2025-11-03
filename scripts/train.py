#!/usr/bin/env python3
"""Dispatcher that routes to the appropriate Cell ARC training pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]

LOG = logging.getLogger(__name__)

TRAINING_MODES: Dict[str, str] = {
    "incontext": "scripts.train_incontext",
    "embedding": "scripts.train_embedding",
}

EMBEDDING_ARCHITECTURES = {
    "tiny_recursive",
    "trm",
    "hrm",
    "transformer_act",
}


def _resolve_training_module(mode: str):
    try:
        module_path = TRAINING_MODES[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unknown training mode '{mode}'. "
            f"Available modes: {sorted(TRAINING_MODES)}"
        ) from exc

    module = __import__(module_path, fromlist=["run"])
    if not hasattr(module, "run"):
        raise AttributeError(
            f"Training module '{module_path}' does not expose a 'run' function."
        )
    return module


@hydra.main(config_path="../configs", config_name="train/default", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    LOG.info("Active repository root: %s", REPO_ROOT)
    LOG.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    training_cfg = cfg.get("training")
    explicit_mode = None
    if training_cfg is not None and training_cfg.get("mode") is not None:
        explicit_mode = str(training_cfg.mode)

    architecture = None
    model_cfg = cfg.get("model")
    if model_cfg is not None and model_cfg.get("architecture") is not None:
        architecture = str(model_cfg.architecture)

    default_mode = (
        "embedding" if architecture in EMBEDDING_ARCHITECTURES else "incontext"
    )
    mode = explicit_mode or default_mode
    if explicit_mode is None and architecture in EMBEDDING_ARCHITECTURES:
        LOG.info(
            "Auto-selecting embedding training mode for recursive architecture '%s'.",
            architecture,
        )
    if explicit_mode is not None and explicit_mode != default_mode:
        LOG.info(
            "Using explicitly configured training mode '%s' for architecture '%s'.",
            explicit_mode,
            architecture,
        )
    LOG.info("Selected training mode: %s", mode)

    module = _resolve_training_module(mode)
    module.run(cfg)


if __name__ == "__main__":
    main()
