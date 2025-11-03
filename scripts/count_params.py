#!/usr/bin/env python3
"""Utility to report parameter counts for each baseline model size profile."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import torch
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines import BaselineConfig, create_baseline, get_baseline_registry

SIZE_CONFIG_DIR = REPO_ROOT / "configs" / "model" / "size"
ALIAS_ARCHITECTURES: set[str] = {"trm"}

# These defaults match the values in configs/train/default.yaml.
DEFAULT_INPUT_VOCAB_SIZE = 17
DEFAULT_OUTPUT_VOCAB_SIZE = 17
DEFAULT_MAX_SEQ_LEN = 272
DEFAULT_BATCH_SIZE = 64


def load_size_variants(size_name: str) -> Mapping[str, Mapping[str, Any]]:
    """Load the architecture-specific overrides for a given size preset."""
    config_path = SIZE_CONFIG_DIR / f"{size_name}.yaml"
    if not config_path.exists():
        raise ValueError(f"Unknown size preset '{size_name}'. Expected file at {config_path}.")
    cfg = OmegaConf.load(config_path)
    variants = cfg.get("variants", {}) or {}
    return OmegaConf.to_container(variants, resolve=True)  # type: ignore[return-value]


def iter_target_architectures(
    requested: Iterable[str] | None,
    available_variants: Mapping[str, Mapping[str, Any]],
) -> Iterable[str]:
    """Return the list of architectures to inspect in order."""
    registry = get_baseline_registry()
    if requested:
        for name in requested:
            if name not in registry:
                raise ValueError(f"Unknown architecture '{name}'. Available: {sorted(registry)}")
        return requested
    # Respect the size config ordering while skipping duplicates like aliases.
    seen: set[str] = set()
    ordered_archs: list[str] = []
    for name in available_variants:
        if name in registry and name not in seen:
            ordered_archs.append(name)
            seen.add(name)
    # Include any remaining registry entries that were not in the size config.
    for name in sorted(registry):
        if name not in seen and name not in ALIAS_ARCHITECTURES:
            ordered_archs.append(name)
            seen.add(name)
    return ordered_archs


def count_parameters_for_architecture(
    architecture: str,
    *,
    model_kwargs: Mapping[str, Any],
) -> int:
    """Instantiate a baseline model and return the total number of parameters."""
    baseline_config = BaselineConfig(
        input_vocab_size=DEFAULT_INPUT_VOCAB_SIZE,
        output_vocab_size=DEFAULT_OUTPUT_VOCAB_SIZE,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
        batch_size=DEFAULT_BATCH_SIZE,
        device=torch.device("cpu"),
        dtype=torch.float32,
        model_kwargs=model_kwargs,
    )
    model = create_baseline(architecture, baseline_config)
    try:
        return sum(param.numel() for param in model.parameters())
    finally:
        # Help garbage collection for larger models.
        del model
        torch.cuda.empty_cache()


def human_readable(count: int) -> str:
    """Format parameter counts in a readable manner."""
    if count >= 1_000_000_000:
        return f"{count:,} ({count / 1_000_000_000:.2f}B)"
    if count >= 1_000_000:
        return f"{count:,} ({count / 1_000_000:.2f}M)"
    if count >= 1_000:
        return f"{count:,} ({count / 1_000:.2f}K)"
    return f"{count:,}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["small", "medium", "large"],
        help="Size presets to evaluate (default: small medium large).",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=None,
        help="Optional subset of architectures to evaluate.",
    )
    args = parser.parse_args()

    for size_name in args.sizes:
        variants = load_size_variants(size_name)
        architectures = iter_target_architectures(args.architectures, variants)
        print(f"\n=== {size_name.upper()} ===")
        for architecture in architectures:
            kwargs = variants.get(architecture, {}) or {}
            try:
                param_count = count_parameters_for_architecture(
                    architecture,
                    model_kwargs=kwargs,
                )
                print(f"{architecture:>15}: {human_readable(param_count)}")
            except Exception as exc:
                print(f"{architecture:>15}: ERROR ({exc})")


if __name__ == "__main__":
    main()
