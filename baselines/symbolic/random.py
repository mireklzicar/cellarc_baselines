from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from .utils import iter_symbols

_NON_SEQUENCE_TYPES = (str, bytes, bytearray)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, _NON_SEQUENCE_TYPES)


class RandomSolver:
    """
    Solver that samples each output position uniformly from the observed vocabulary.

    The vocabulary consists of every integer observed in the episode's training pairs
    (inputs and outputs) plus the query itself. When no symbols are observed the solver
    falls back to emitting ``default_symbol``.
    """

    def __init__(self, *, seed: int | None = 0, default_symbol: int = 0) -> None:
        self._rng = random.Random(seed)
        self._default_symbol = int(default_symbol)

    def predict(self, episode: Mapping[str, Any]) -> Any:
        if "query" not in episode:
            raise KeyError("Episode dictionary must contain a 'query' field.")

        vocabulary = self._collect_vocabulary(episode)
        if not vocabulary:
            vocabulary = [self._default_symbol]

        return self._sample_like(episode["query"], vocabulary)

    def _collect_vocabulary(self, episode: Mapping[str, Any]) -> list[int]:
        observed: list[int] = []
        seen: set[int] = set()

        def _add(symbol: int | None) -> None:
            if symbol is None:
                return
            value = int(symbol)
            if value not in seen:
                seen.add(value)
                observed.append(value)

        for pair in episode.get("train", []):
            for symbol in iter_symbols(pair.get("input", [])):
                _add(symbol)
            for symbol in iter_symbols(pair.get("output", [])):
                _add(symbol)

        for symbol in iter_symbols(episode.get("query", [])):
            _add(symbol)

        return observed

    def _sample_like(self, template: Any, vocabulary: list[int]) -> Any:
        if not _is_sequence(template):
            return int(self._rng.choice(vocabulary))
        return [self._sample_like(item, vocabulary) for item in template]


__all__ = ["RandomSolver"]
