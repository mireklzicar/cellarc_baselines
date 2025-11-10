#!/usr/bin/env python3
"""Download train/val loss curves for all W&B runs and save a single CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import wandb

ENTITY = "lzicar"
PROJECTS = [
    "cellarc100k_50e_embedding_small",
    "cellarc100k_50e_incontext_medium",
    "cellarc100k_50e_embedding_large",
    "cellarc100k_50e_incontext_small",
    "cellarc100k_50e_incontext_large",
    "cellarc100k_50e_embedding_medium",
]
TRAIN_LOSS_KEYS = ("train/loss", "train/loss_mean")
VAL_LOSS_KEYS = ("val/loss/val", "val/loss_mean", "val/loss")


def parse_training_metadata(project: str) -> Tuple[Optional[str], Optional[str]]:
    """Infer training mode and size from the project name."""
    parts = project.split("_")
    if len(parts) < 2:
        return None, None
    mode_raw = parts[-2].lower()
    size = parts[-1].lower()
    mode = "in context" if mode_raw == "incontext" else mode_raw
    return mode, size


def first_matching_value(row: Dict[str, Any], keys: Sequence[str], substring: str) -> Optional[float]:
    """Return the first numeric value that matches a key or substring."""
    for key in keys:
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, (int, float)):
                return float(value)
    for key, value in row.items():
        if substring in key and isinstance(value, (int, float)):
            return float(value)
    return None


def collect_run_loss_history(
    run: wandb.apis.public.Run,
    *,
    project: str,
    training_mode: Optional[str],
    size: Optional[str],
) -> List[Dict[str, Any]]:
    """Collect per-step train/val loss for a single run."""
    records: List[Dict[str, Any]] = []
    history_iter: Iterable[Dict[str, Any]] = run.scan_history(page_size=1000)

    for row in history_iter:
        step = row.get("_step")
        if step is None:
            continue
        train_loss = first_matching_value(row, TRAIN_LOSS_KEYS, "train/loss")
        val_loss = first_matching_value(row, VAL_LOSS_KEYS, "val/loss")
        if train_loss is None and val_loss is None:
            continue

        records.append(
            {
                "project": project,
                "training_mode": training_mode,
                "size": size,
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "url": run.url,
                "step": step,
                "timestamp": row.get("_timestamp"),
                "runtime": row.get("_runtime"),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
    return records


def collect_all_loss_curves(entity: str, projects: Sequence[str]) -> pd.DataFrame:
    """Fetch every run's loss history for the provided projects."""
    api = wandb.Api()
    all_records: List[Dict[str, Any]] = []

    for project in projects:
        project_path = f"{entity}/{project}"
        print(f"Fetching runs for {project_path} ...")
        training_mode, size = parse_training_metadata(project)
        try:
            runs = api.runs(project_path)
        except wandb.errors.CommError as exc:  # type: ignore[attr-defined]
            print(f"Failed to list runs for {project_path}: {exc}")
            continue

        for run in runs:
            print(f"  Collecting history for run {run.id} ({run.name})")
            try:
                records = collect_run_loss_history(
                    run,
                    project=project,
                    training_mode=training_mode,
                    size=size,
                )
                all_records.extend(records)
            except wandb.errors.CommError as exc:  # type: ignore[attr-defined]
                print(f"    Failed to download history for {run.id}: {exc}")

    df = pd.DataFrame(all_records)
    if not df.empty:
        df = df.sort_values(["project", "run_name", "step"], ignore_index=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch train/val loss curves for all W&B runs.")
    parser.add_argument(
        "--entity",
        default=ENTITY,
        help=f"W&B entity/organization (default: {ENTITY}).",
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        default=PROJECTS,
        help="W&B project names to include (default: all cellarc projects).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("wandb_export/all_wandb_loss_curves.csv"),
        help="Destination CSV file (default: wandb_export/all_wandb_loss_curves.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = collect_all_loss_curves(args.entity, args.projects)
    if df.empty:
        print("No loss metrics found for the requested projects.")
        return

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
