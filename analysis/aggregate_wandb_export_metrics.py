#!/usr/bin/env python3
"""
Aggregate runtime, batch size, accuracy, parameter, and metadata information from wandb_export files.

The script scans all *.json exports in wandb_export/, extracts the requested metrics,
and prints a single concatenated pandas DataFrame (optionally saving it to CSV).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd


def iter_key_values(obj: Any, prefix: str = "") -> Iterator[Tuple[str, Any]]:
    """Yield flattened key/value pairs for arbitrarily nested dict/list structures."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_key_values(value, new_prefix)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            new_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from iter_key_values(value, new_prefix)
    else:
        yield prefix, obj


def coerce_numeric(value: Any) -> Optional[float]:
    """Convert a value to float when possible, handling plain numbers and unit suffixes."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        for suffix in ("gb", "mb", "kb"):
            if cleaned.lower().endswith(suffix):
                cleaned = cleaned[: -len(suffix)]
                break
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def extract_batch_size(row: Dict[str, Any]) -> Optional[float]:
    """Search config sections for the first batch size-like field."""
    config_subset = {k: v for k, v in row.items() if isinstance(k, str) and k.startswith("config")}
    for key, value in iter_key_values(config_subset):
        key_lower = key.lower()
        if "batch" in key_lower:
            numeric = coerce_numeric(value)
            if numeric is not None:
                return numeric
    return None


def parse_training_metadata(project: str) -> Tuple[Optional[str], Optional[str]]:
    """Infer training mode and size from the project name."""
    parts = project.split("_")
    if len(parts) < 2:
        return None, None
    mode_raw = parts[-2].lower()
    size = parts[-1].lower()
    if mode_raw == "incontext":
        mode = "in context"
    else:
        mode = mode_raw
    return mode, size


def load_runs(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open() as fp:
        return json.load(fp)


def build_dataframe(wandb_dir: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for json_path in sorted(wandb_dir.glob("*.json")):
        runs = load_runs(json_path)
        project = json_path.stem
        training_mode, size = parse_training_metadata(project)
        for row in runs:
            records.append(
                {
                    "project": project,
                    "training_mode": training_mode,
                    "size": size,
                    "run_id": row.get("run_id"),
                    "run_name": row.get("name"),
                    "state": row.get("state"),
                    "created_at": row.get("created_at"),
                    "runtime_seconds": row.get("summary._runtime"),
                    "batch_size": extract_batch_size(row),
                    "test_extrapolation_accuracy": row.get("summary.test/accuracy/test_extrapolation"),
                    "test_interpolation_accuracy": row.get("summary.test/accuracy/test_interpolation"),
                    "num_parameters": row.get("summary.model/num_parameters"),
                }
            )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["project", "run_name"], ignore_index=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate wandb_export runtimes, batch sizes, accuracies, parameter counts, and metadata."
    )
    parser.add_argument(
        "--wandb-dir",
        type=Path,
        default=Path("wandb_export"),
        help="Directory that holds exported project JSON files (default: wandb_export).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("wandb_export/all_wandb_runs_summary.csv"),
        help="CSV path for aggregated dataframe (default: wandb_export/all_wandb_runs_summary.csv).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("wandb_export/all_wandb_runs_summary.json"),
        help="JSON path for aggregated dataframe (default: wandb_export/all_wandb_runs_summary.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.wandb_dir.exists():
        raise SystemExit(f"{args.wandb_dir} does not exist.")

    df = build_dataframe(args.wandb_dir)
    if df.empty:
        print("No runs found in the provided wandb_export directory.")
        return

    print(df.to_string(index=False))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    df.to_json(args.output_json, orient="records", indent=2)
    print(f"\nWrote aggregated results to {args.output_csv} and {args.output_json}")


if __name__ == "__main__":
    main()
