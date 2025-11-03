from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

_NON_SEQUENCE_TYPES = (str, bytes, bytearray)


def _is_sequence(obj: Any) -> bool:
    return isinstance(obj, Sequence) and not isinstance(obj, _NON_SEQUENCE_TYPES)


def clone_as_ints(value: Any) -> Any:
    """Return a deep copy of ``value`` with every element cast to ``int``."""
    if not _is_sequence(value):
        if value is None:
            raise ValueError("Encountered None while cloning; expected integers.")
        return int(value)
    return [clone_as_ints(item) for item in value]


def fill_like(template: Any, fill_value: int) -> Any:
    """Produce a structure with the same shape as ``template`` filled with ``fill_value``."""
    if not _is_sequence(template):
        return int(fill_value)
    return [fill_like(item, fill_value) for item in template]


def iter_symbols(value: Any) -> Iterator[int]:
    """Yield every integer observed in ``value``."""
    if not _is_sequence(value):
        if value is None:
            return
        yield int(value)
        return
    for item in value:
        yield from iter_symbols(item)
