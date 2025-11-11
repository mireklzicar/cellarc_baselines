#!/usr/bin/env python3
"""Compare per-task accuracies between two models and plot their correlation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt


def _load_per_task_map(directory: Path, model_name: str, split: str) -> Dict[str, float]:
    file_path = directory / f"{model_name}__{split}_per_task_accuracy.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing per-task file: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("per_task_accuracy")
    if not isinstance(entries, list):
        raise ValueError(f"Unexpected payload structure in {file_path}")
    mapping: Dict[str, float] = {}
    for entry in entries:
        episode_id = entry.get("episode_id")
        token_accuracy = entry.get("token_accuracy")
        if episode_id is None or token_accuracy is None:
            continue
        mapping[str(episode_id)] = float(token_accuracy)
    return mapping


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("Sequences must have the same length for Pearson correlation.")
    n = len(xs)
    if n < 2:
        return float("nan")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return float("nan")
    return cov / (var_x**0.5 * var_y**0.5)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-task accuracy correlation between two models.")
    parser.add_argument("--model-a-name", required=True, help="Name/prefix of model A (e.g., gpt-5-...).")
    parser.add_argument("--model-a-dir", required=True, help="Directory containing model A per-task files.")
    parser.add_argument("--model-b-name", required=True, help="Name/prefix of model B (e.g., de_bruijn).")
    parser.add_argument("--model-b-dir", required=True, help="Directory containing model B per-task files.")
    parser.add_argument(
        "--splits",
        nargs="+",
        required=True,
        help="One or more split names to compare (must exist for both models).",
    )
    parser.add_argument("--output-plot", required=True, help="Path to the scatter plot image to write.")
    parser.add_argument(
        "--output-data",
        default=None,
        help="Optional CSV filepath to store the merged per-task accuracy table.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output resolution (dots per inch) for the scatter plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_a_dir = Path(args.model_a_dir).expanduser().resolve()
    model_b_dir = Path(args.model_b_dir).expanduser().resolve()
    output_plot = Path(args.output_plot).expanduser().resolve()
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    merged_rows: List[dict[str, object]] = []
    for split in args.splits:
        a_map = _load_per_task_map(model_a_dir, args.model_a_name, split)
        b_map = _load_per_task_map(model_b_dir, args.model_b_name, split)
        common_ids = sorted(set(a_map) & set(b_map))
        if not common_ids:
            continue
        for episode_id in common_ids:
            merged_rows.append(
                {
                    "split": split,
                    "episode_id": episode_id,
                    "model_a": a_map[episode_id],
                    "model_b": b_map[episode_id],
                }
            )

    if not merged_rows:
        raise SystemExit("No overlapping episodes found between the provided models.")

    xs = [float(row["model_b"]) for row in merged_rows]
    ys = [float(row["model_a"]) for row in merged_rows]
    pearson_r = _pearson(xs, ys)

    grouped: Dict[str, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    for row in merged_rows:
        x_list, y_list = grouped[row["split"]]
        x_list.append(float(row["model_b"]))
        y_list.append(float(row["model_a"]))

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e"])
    color_cycle = iter(colors)
    color_map: Dict[str, str] = {}

    for split in args.splits:
        if split not in grouped:
            continue
        color_map.setdefault(split, next(color_cycle, "#333333"))
        x_values, y_values = grouped[split]
        ax.scatter(
            x_values,
            y_values,
            label=split,
            alpha=0.8,
            s=30,
            color=color_map[split],
            edgecolors="none",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"{args.model_b_name} token accuracy")
    ax.set_ylabel(f"{args.model_a_name} token accuracy")
    ax.set_title(
        f"Per-task token accuracy correlation\nPearson r = {pearson_r:.3f}" if pearson_r == pearson_r else "Per-task token accuracy correlation"
    )
    if len(grouped) > 1:
        ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=args.dpi)
    print(f"Wrote scatter plot to {output_plot}")

    if args.output_data:
        output_data_path = Path(args.output_data).expanduser().resolve()
        output_data_path.parent.mkdir(parents=True, exist_ok=True)
        with output_data_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["episode_id", "split", args.model_b_name, args.model_a_name],
            )
            writer.writeheader()
            for row in merged_rows:
                writer.writerow(
                    {
                        "episode_id": row["episode_id"],
                        "split": row["split"],
                        args.model_b_name: f"{row['model_b']:.6f}",
                        args.model_a_name: f"{row['model_a']:.6f}",
                    }
                )
        print(f"Wrote merged per-task table to {output_data_path}")


if __name__ == "__main__":
    main()
