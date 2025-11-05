#!/usr/bin/env python3
"""Generate parameter counts for the smoke-test runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Iterable, List, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.count_params import (  # noqa: E402
    count_parameters_for_architecture,
    load_size_variants,
)

ARCHES: Sequence[str] = (
    "rnn",
    "rnn_ar",
    "transformer",
    "transformer_ar",
    "transformer_act",
    "cnn1d",
    "nca1d",
    "tiny_recursive",
    "hrm",
)

EMBEDDING_ARCHES: set[str] = {
    "rnn",
    "rnn_ar",
    "transformer",
    "transformer_ar",
    "transformer_act",
    "cnn1d",
    "nca1d",
    "tiny_recursive",
    "hrm",
}

MODES: Sequence[str] = ("incontext", "embedding")


def iter_run_specs(sizes: Iterable[str]) -> Iterable[tuple[str, str, str, Mapping[str, object]]]:
    """Yield (project, run_name, architecture, kwargs) tuples for each run."""
    for size in sizes:
        size_variants = load_size_variants(size)
        for mode in MODES:
            project_name = f"cellarc100k_{mode}_baselines_{size}"
            for arch in ARCHES:
                if mode == "embedding" and arch not in EMBEDDING_ARCHES:
                    continue
                model_run_name = f"{arch}_{size}_{mode}"
                kwargs = size_variants.get(arch, {}) or {}
                yield project_name, model_run_name, arch, kwargs


def compute_param_rows(sizes: Iterable[str]) -> List[tuple[str, str, int]]:
    """Instantiate models and collect parameter counts."""
    rows: List[tuple[str, str, int]] = []
    for project, run_name, arch, kwargs in iter_run_specs(sizes):
        param_count = count_parameters_for_architecture(arch, model_kwargs=kwargs)
        rows.append((project, run_name, param_count))
    # Clear CUDA caches once at the end to avoid lingering allocations.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def write_csv(rows: Sequence[tuple[str, str, int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("project_name", "model_run_name", "num_params"))
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["small", "medium", "large"],
        help="Size presets to evaluate (default: medium).",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "smoke_param_counts.csv"),
        help="Destination CSV path (default: outputs/smoke_param_counts.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = compute_param_rows(args.sizes)
    output_path = Path(args.output)
    write_csv(rows, output_path)
    for project, run_name, count in rows:
        print(f"{project},{run_name},{count}")
    print(f"\nWrote {len(rows)} entries to {output_path}")


if __name__ == "__main__":
    main()
