from __future__ import annotations

from typing import Any, Mapping

from .utils import clone_as_ints


class CopycatSolver:
    """Solver that copies the query grid verbatim."""

    def predict(self, episode: Mapping[str, Any]) -> Any:
        if "query" not in episode:
            raise KeyError("Episode dictionary must contain a 'query' field.")
        return clone_as_ints(episode["query"])


__all__ = ["CopycatSolver"]
