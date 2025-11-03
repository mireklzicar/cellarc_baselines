from __future__ import annotations

import random
from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Symbol = int
Word = List[Symbol]
Window = Tuple[Symbol, ...]


def _centered_window(seq: Word, i: int, W: int, wrap: bool) -> Window:
    """Return the length-W window centered at i (half = (W-1)//2)."""
    n = len(seq)
    half = W // 2
    if wrap:
        return tuple(seq[(i - half + j) % n] for j in range(W))
    out: List[Symbol] = []
    for j in range(i - half, i + half + 1):
        if 0 <= j < n:
            out.append(seq[j])
        else:
            out.append(0)
    return tuple(out)


@dataclass
class LearnedLocalMap:
    W: int
    wrap: bool
    mapping: Dict[Window, Symbol]
    alphabet_size: Optional[int] = None
    windows_total: Optional[int] = None
    alphabet: Optional[Sequence[Symbol]] = None
    rng_seed: Optional[int] = None

    @property
    def coverage_complete(self) -> bool:
        if self.alphabet_size is None or self.windows_total is None:
            return False
        return len(self.mapping) == self.windows_total == (self.alphabet_size ** self.W)

    def _alphabet(self) -> List[Symbol]:
        if self.alphabet is not None:
            return list(self.alphabet)
        if self.alphabet_size is not None:
            return list(range(self.alphabet_size))
        observed = sorted(set(self.mapping.values()))
        if observed:
            return observed
        return [0]

    def _majority_symbol(self) -> Symbol:
        if not self.mapping:
            return 0
        from collections import Counter

        return Counter(self.mapping.values()).most_common(1)[0][0]

    def predict(
        self,
        query: Word,
        *,
        rng: Optional[random.Random] = None,
        fallback: str = "random",
        default: int = 0,
    ) -> Word:
        """
        Predict the automaton response for ``query``.

        Parameters
        ----------
        query:
            Input symbols to evaluate.
        rng:
            Optional random generator used when ``fallback='random'``.
        fallback:
            Strategy for unseen windows. Supported values:
              - ``'random'``: sample uniformly from the alphabet (default).
              - ``'majority'``: reuse the most frequent observed output.
              - ``'default'``: emit the provided ``default`` symbol.
              - ``'strict'``: raise ``KeyError`` as soon as a window is missing.
        default:
            Symbol to emit when ``fallback='default'``.
        """
        if fallback not in {"random", "majority", "default", "strict"}:
            raise ValueError("fallback must be one of {'random','majority','default','strict'}")

        rng_local = rng
        if fallback == "random":
            if rng_local is None:
                seed = self.rng_seed if self.rng_seed is not None else 0
                rng_local = random.Random(seed)
            alphabet = self._alphabet()
            random_cache: Dict[Window, Symbol] = {}
        else:
            alphabet = []
            random_cache = {}

        if fallback == "majority":
            mode_symbol = self._majority_symbol()
        elif fallback == "default":
            mode_symbol = default
        else:
            mode_symbol = 0

        missing: List[Window] = []
        n = len(query)
        out: List[Symbol] = [0] * n
        for i in range(n):
            win = _centered_window(query, i, self.W, self.wrap)
            val = self.mapping.get(win)
            if val is not None:
                out[i] = val
                continue

            if fallback == "random":
                cached = random_cache.get(win)
                if cached is None:
                    cached = rng_local.choice(alphabet)  # type: ignore[arg-type]
                    random_cache[win] = cached
                out[i] = cached
            elif fallback == "majority":
                out[i] = mode_symbol
            elif fallback == "default":
                out[i] = mode_symbol
            else:
                missing.append(win)

        if missing and fallback == "strict":
            ex = missing[0]
            raise KeyError(
                f"Missing {len(missing)} of {self.alphabet_size ** self.W if self.alphabet_size else '??'} "
                f"windows; e.g. first unseen window={ex}"
            )
        return out

    def predict_with_backoff(self, query: Word, policy: str = "majority", default: int = 0) -> Word:
        """
        Predict while handling unseen windows via a simple backoff policy:
          - 'majority': emit the most frequent symbol observed in training.
          - 'default':  always emit the supplied default symbol.
          - 'random':   guess uniformly from the alphabet using the stored RNG seed.
        """
        if policy not in {"majority", "default", "random"}:
            raise ValueError("policy must be 'majority', 'default', or 'random'")

        if policy == "majority":
            return self.predict(query, fallback="majority")
        if policy == "default":
            return self.predict(query, fallback="default", default=default)
        return self.predict(query, fallback="random")


def learn_local_map_from_pairs(
    train_pairs: Iterable[Tuple[Word, Word]],
    W: int,
    wrap: bool = True,
    alphabet_size: Optional[int] = None,
    windows_total: Optional[int] = None,
) -> LearnedLocalMap:
    """
    Learn F: (length-W window) -> output, using only *interior* indices so we never
    depend on per-segment wrap. With context-added segments, this recovers all k**W windows.
    """
    table: Dict[Window, Symbol] = {}
    half = W // 2
    for x, y in train_pairs:
        if len(x) != len(y):
            raise ValueError("Input/output lengths in a training pair must match.")
        n = len(x)
        for i in range(half, n - half):
            win = _centered_window(x, i, W, wrap=True)
            val = y[i]
            prev = table.get(win)
            if prev is None:
                table[win] = val
            elif prev != val:
                raise ValueError(f"Inconsistent mapping for window {win}: saw {prev} then {val}.")
    return LearnedLocalMap(
        W=W,
        wrap=wrap,
        mapping=table,
        alphabet_size=alphabet_size,
        windows_total=windows_total,
        alphabet=None,
    )


def _infer_wrap_from_record(rec: dict) -> bool:
    try:
        meta = rec.get("meta", {})
        if "wrap" in meta:
            return bool(meta["wrap"])
    except Exception:
        pass
    try:
        return bool(rec["program"]["ca"]["wrap"])
    except Exception:
        return True


def _infer_window_from_pairs(
    train_pairs: Sequence[Tuple[Word, Word]],
    *,
    wrap: bool,
) -> int:
    if not train_pairs:
        return 1

    max_len = max(len(x) for x, _y in train_pairs)
    if max_len <= 0:
        return 1

    upper = min(_MAX_INFER_WINDOW, max_len)
    candidates = [w for w in range(upper, 0, -1) if w % 2 == 1]
    if not candidates:
        candidates = [1]

    for window in candidates:
        try:
            learn_local_map_from_pairs(train_pairs, W=window, wrap=wrap)
        except ValueError:
            continue
        LOGGER.debug("Inferred CA window %s from training pairs.", window)
        return window

    LOGGER.debug("Falling back to window size 1 for DeBruijnSolver.")
    return 1


def learn_from_record(rec: dict) -> LearnedLocalMap:
    meta_raw = rec.get("meta")
    meta = dict(meta_raw) if isinstance(meta_raw, Mapping) else {}

    wrap = _infer_wrap_from_record(rec)

    train_pairs = [(pair["input"], pair["output"]) for pair in rec["train"]]
    W_meta = meta.get("window")
    if W_meta is not None:
        W = int(W_meta)
    else:
        W = _infer_window_from_pairs(train_pairs, wrap=wrap)

    alpha_size_meta = meta.get("alphabet_size")
    k = int(alpha_size_meta) if alpha_size_meta is not None else None
    windows_total_meta = meta.get("windows_total")
    windows_total = int(windows_total_meta) if windows_total_meta is not None else None

    alphabet: Optional[Sequence[Symbol]]
    if k is not None:
        alphabet = tuple(range(k))
    else:
        observed: List[Symbol] = []
        for x, y in train_pairs:
            observed.extend(x)
            observed.extend(y)
        unique = sorted(set(observed))
        alphabet = tuple(unique) if unique else None
        if unique:
            k = len(unique)

    try:
        episode_seed = int(meta.get("episode_seed")) if "episode_seed" in meta else None
    except Exception:
        episode_seed = None
    model = learn_local_map_from_pairs(
        train_pairs=train_pairs,
        W=W,
        wrap=wrap,
        alphabet_size=k,
        windows_total=windows_total,
    )
    model.alphabet = alphabet
    model.rng_seed = episode_seed
    return model


class DeBruijnSolver:
    """Episode-level wrapper around the De Bruijn local map learner."""

    _SUPPORTED_FALLBACKS = {"majority", "default", "random", "strict"}

    def __init__(self, *, fallback: str = "majority", default_symbol: int = 0) -> None:
        fallback_normalized = fallback.lower()
        if fallback_normalized not in self._SUPPORTED_FALLBACKS:
            raise ValueError(
                f"Unsupported fallback policy {fallback!r}. "
                f"Valid options: {sorted(self._SUPPORTED_FALLBACKS)}"
            )
        self._fallback = fallback_normalized
        self._default_symbol = int(default_symbol)

    def predict(self, episode: Mapping[str, Any]) -> List[int]:
        if "query" not in episode:
            raise KeyError("Episode dictionary must contain a 'query' field.")

        model = learn_from_record(episode)
        query = [int(value) for value in episode["query"]]

        if self._fallback == "strict":
            return model.predict(query, fallback="strict")
        if self._fallback == "default":
            return model.predict_with_backoff(query, policy="default", default=self._default_symbol)
        if self._fallback == "random":
            return model.predict_with_backoff(query, policy="random")
        return model.predict_with_backoff(query, policy="majority")


__all__ = [
    "DeBruijnSolver",
    "LearnedLocalMap",
    "learn_from_record",
    "learn_local_map_from_pairs",
]
LOGGER = logging.getLogger(__name__)

_MAX_INFER_WINDOW = 15
