#!/usr/bin/env python3
"""Generate full LLM prompts for a few benchmark episodes for inspection."""

from __future__ import annotations

import argparse
import sys
from itertools import islice
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from cellarc import download_benchmark
from baselines.llm.prompting import PromptBuilder, PromptBuilderConfig
from baselines.llm.runner import _load_dataset_with_fallback  # reuse eval loader

DEFAULT_DATASET_ROOT = REPO_ROOT / "outputs" / "hf_datasets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/llm/default.yaml",
        help="Path to the Hydra config used for LLM evaluation.",
    )
    parser.add_argument(
        "--split",
        default="test_interpolation_100",
        help="Dataset split to preview (e.g., test_interpolation_100).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of episode prompts to print.",
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional dataset root override. Defaults to the path in the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(resolve_path(args.config))

    instructions_path = resolve_path(cfg.prompt.instructions_path)
    instructions_text = read_text(instructions_path)

    builder = PromptBuilder(
        PromptBuilderConfig(
            instructions=instructions_text,
            system_message=cfg.prompt.system_message,
            response_format_hint=cfg.prompt.response_format_hint,
            max_train_examples=cfg.prompt.max_train_examples,
            include_episode_id=cfg.prompt.include_episode_id,
        )
    )

    cfg.dataset.root = prepare_dataset_root(args.dataset_root or cfg.dataset.root)
    ensure_dataset_available(cfg)
    dataset = _load_dataset_with_fallback(cfg, args.split)

    print(
        f"[preview_prompts] instructions={instructions_path} "
        f"split={args.split} count={args.count}"
    )
    print("")

    for idx, (dataset_index, episode) in enumerate(islice(enumerate(dataset), args.count), start=1):
        prompt = builder.build_prompt(episode)
        episode_id = episode.get("id")
        print("=" * 80)
        header = f"Episode #{idx} (dataset index {dataset_index}"
        if episode_id:
            header += f", id={episode_id}"
        header += ")"
        print(header)
        print("-" * 80)
        if prompt.system:
            print("System Message:")
            print(prompt.system)
            print("")
        print("User Prompt:")
        print(prompt.user)
        print("=" * 80)
        print("")


def resolve_path(path_str: str | None) -> Path:
    if not path_str:
        raise ValueError("Expected a valid path string.")
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def prepare_dataset_root(root_option: str | None) -> str:
    if root_option:
        root_path = Path(root_option).expanduser().resolve()
    else:
        root_path = DEFAULT_DATASET_ROOT
    root_path.mkdir(parents=True, exist_ok=True)
    return str(root_path)


def ensure_dataset_available(cfg) -> None:
    root_path = Path(cfg.dataset.root).expanduser().resolve()
    download_benchmark(
        name=cfg.dataset.benchmark,
        include_metadata=cfg.dataset.include_metadata,
        root=root_path,
        force_download=False,
    )
    cfg.dataset.root = str(root_path)


if __name__ == "__main__":
    main()
