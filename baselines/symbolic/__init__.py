from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Mapping, Protocol

from .copycat import CopycatSolver
from .de_bruijn_solver import DeBruijnSolver
from .most_frequent import MostFrequentSolver
from .random import RandomSolver


class SymbolicSolver(Protocol):
    """Protocol describing the interface for symbolic baselines."""

    def predict(self, episode: Mapping[str, Any]) -> Any:
        ...


SymbolicSolverFactory = Callable[..., SymbolicSolver]

_SYMBOLIC_BASELINES: dict[str, SymbolicSolverFactory] = {
    "copycat": CopycatSolver,
    "most_frequent": MostFrequentSolver,
    "de_bruijn": DeBruijnSolver,
    "random": RandomSolver,
}

LOGGER = logging.getLogger(__name__)


def _filter_kwargs(factory: SymbolicSolverFactory, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    if not kwargs:
        return {}
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return {k: v for k, v in kwargs.items() if v is not None}

    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return {k: v for k, v in kwargs.items() if v is not None}

    valid_names = {
        param.name
        for param in signature.parameters.values()
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and param.name != "self"
    }

    filtered = {k: v for k, v in kwargs.items() if k in valid_names and v is not None}
    dropped = sorted(k for k in kwargs.keys() if k not in filtered and kwargs[k] is not None)
    if dropped:
        LOGGER.debug("Dropping unused kwargs %s for %s", dropped, factory)
    return filtered


def get_symbolic_registry() -> dict[str, SymbolicSolverFactory]:
    """Return a read-only snapshot of symbolic baseline factories."""
    return dict(_SYMBOLIC_BASELINES)


def create_symbolic_baseline(name: str, **kwargs: Any) -> SymbolicSolver:
    try:
        factory = _SYMBOLIC_BASELINES[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown symbolic baseline {name!r}. Available: {list(_SYMBOLIC_BASELINES)}"
        ) from exc
    filtered_kwargs = _filter_kwargs(factory, kwargs)
    return factory(**filtered_kwargs)


__all__ = [
    "CopycatSolver",
    "DeBruijnSolver",
    "MostFrequentSolver",
    "RandomSolver",
    "SymbolicSolver",
    "SymbolicSolverFactory",
    "create_symbolic_baseline",
    "get_symbolic_registry",
]
