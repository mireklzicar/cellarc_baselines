from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from .utils import fill_like, iter_symbols


class MostFrequentSolver:
    """Solver that emits the most common colour observed in the episode."""

    def __init__(self, *, default_symbol: int = 0) -> None:
        self._default_symbol = int(default_symbol)

    def predict(self, episode: Mapping[str, Any]) -> Any:
        if "query" not in episode:
            raise KeyError("Episode dictionary must contain a 'query' field.")

        counts: Counter[int] = Counter()
        for pair in episode.get("train", []):
            counts.update(iter_symbols(pair.get("input", [])))
            counts.update(iter_symbols(pair.get("output", [])))

        counts.update(iter_symbols(episode["query"]))

        if counts:
            most_common = max(counts.items(), key=lambda item: (item[1], -item[0]))[0]
        else:
            most_common = self._default_symbol

        return fill_like(episode["query"], most_common)


__all__ = ["MostFrequentSolver"]
