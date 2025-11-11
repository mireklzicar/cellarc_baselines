#!/usr/bin/env python3
"""Export per-task token accuracies from JSONL prediction logs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, List


def _compute_token_stats(prediction: List[Any], target: List[Any]) -> tuple[int, int]:
    """Return (correct_tokens, total_tokens) for two sequences."""
    total = max(len(prediction), len(target))
    if total == 0:
        return 0, 0
    correct = sum(
        1
        for pred_symbol, target_symbol in zip_longest(prediction, target, fillvalue=None)
        if pred_symbol == target_symbol
    )
    return correct, total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-task token accuracy JSON files for each split."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the JSONL predictions file (one JSON object per line).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to place the per-split JSON files (defaults to input directory).",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional prefix for output filenames (defaults to the input stem).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    filename_prefix = args.prefix or input_path.stem
    per_split: dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to parse line {line_number}: {exc}") from exc

            missing_keys = [key for key in ("split", "episode_id", "parsed_output", "solution") if key not in record]
            if missing_keys:
                raise SystemExit(
                    f"Line {line_number} is missing required keys: {', '.join(missing_keys)}"
                )

            split = str(record["split"])
            episode_id = str(record["episode_id"])
            prediction = record["parsed_output"] or []
            target = record["solution"] or []

            if not isinstance(prediction, list) or not isinstance(target, list):
                raise SystemExit(
                    f"Line {line_number}: 'parsed_output' and 'solution' must be lists."
                )

            correct, total = _compute_token_stats(prediction, target)
            token_accuracy = (correct / total) if total else 0.0
            per_split[split].append(
                {
                    "episode_id": episode_id,
                    "episode_index": record.get("episode_index"),
                    "token_correct": correct,
                    "token_total": total,
                    "token_accuracy": token_accuracy,
                    "solved": prediction == target,
                }
            )

    if not per_split:
        raise SystemExit(f"No prediction records found in {input_path}")

    for split, entries in per_split.items():
        output_path = output_dir / f"{filename_prefix}__{split}_per_task_accuracy.json"
        payload = {
            "split": split,
            "episodes": len(entries),
            "input_path": str(input_path),
            "per_task_accuracy": entries,
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"Wrote {len(entries)} per-task records to {output_path}")


if __name__ == "__main__":
    main()
