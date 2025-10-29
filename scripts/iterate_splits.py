#!/usr/bin/env python3
"""Download Cell ARC splits from the Hugging Face Hub and iterate over them once.

This is a lightweight smoke-test that ensures the remote snapshot is accessible and
that your environment can stream the dataset end-to-end.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from cell_arc import EpisodeDataset
from tqdm.auto import tqdm


def iterate_once(split: str, *, benchmark: str, include_metadata: bool, root: str | None) -> None:
    """Stream a single split while showing a tqdm progress bar."""

    dataset = EpisodeDataset.from_huggingface(
        split=split,
        name=benchmark,
        include_metadata=include_metadata,
        root=root,
    )

    try:
        total = len(dataset)
    except Exception:  # pragma: no cover - jsonl paths fall back to unknown length
        total = None

    for _ in tqdm(dataset, desc=f"{split} episodes", total=total):
        pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterate once through the requested Cell ARC splits with tqdm progress bars.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("train", "val", "test"),
        help="Split names to iterate (default: train val test).",
    )
    parser.add_argument(
        "--benchmark",
        default="cellarc_100k",
        help="Remote benchmark name registered with cell_arc (default: cellarc_100k).",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Optional local cache directory passed to EpisodeDataset.from_huggingface.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip downloading the companion metadata repository.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    include_metadata = not args.no_metadata

    for split in args.splits:
        iterate_once(
            split=split,
            benchmark=args.benchmark,
            include_metadata=include_metadata,
            root=args.root,
        )


if __name__ == "__main__":
    main()
