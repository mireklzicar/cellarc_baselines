"""Iterable dataset that streams single I/O samples for puzzle-embedding training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
import random

import torch
from torch.utils.data import IterableDataset

from scripts.train_incontext import TokenVocabulary


@dataclass
class PuzzleEmbeddingSample:
    """Single training example comprising one input/output pair and its puzzle id."""

    puzzle_identifier: int
    input_ids: List[int]
    target_ids: List[int]
    labels: List[int]
    loss_mask: List[bool]
    sample_type: str = "support"


class PuzzleIdentifierTable:
    """Shared mapping assigning a stable integer id to each puzzle key."""

    def __init__(self) -> None:
        self._mapping: Dict[str, int] = {}
        self._next_id: int = 0

    def get_or_create(self, key: str) -> int:
        """Return the identifier for a puzzle, creating it if necessary."""
        value = self._mapping.get(key)
        if value is None:
            value = self._next_id
            self._mapping[key] = value
            self._next_id += 1
        return value

    def get(self, key: str) -> Optional[int]:
        """Return the identifier for a puzzle without creating a new one."""
        return self._mapping.get(key)

    @property
    def size(self) -> int:
        return self._next_id


def _append_token(
    sample: PuzzleEmbeddingSample,
    token_id: int,
    *,
    tokens: TokenVocabulary,
    ignore_index: int,
    target_id: Optional[int] = None,
    label: Optional[int] = None,
    contributes_to_loss: bool = False,
) -> None:
    sample.input_ids.append(int(token_id))
    sample.target_ids.append(int(target_id) if target_id is not None else tokens.pad_id)
    if label is None:
        sample.labels.append(ignore_index)
    else:
        sample.labels.append(int(label))
    sample.loss_mask.append(bool(contributes_to_loss))


def build_support_sample(
    *,
    puzzle_identifier: int,
    pair: Dict[str, Sequence[int]],
    tokens: TokenVocabulary,
    ignore_index: int,
    max_seq_len: int,
    drop_long: bool,
) -> Optional[PuzzleEmbeddingSample]:
    """Construct a sample for a single input/output training pair."""

    sample = PuzzleEmbeddingSample(
        puzzle_identifier=puzzle_identifier,
        input_ids=[],
        target_ids=[],
        labels=[],
        loss_mask=[],
        sample_type="support",
    )

    _append_token(
        sample,
        tokens.support_input_id,
        tokens=tokens,
        ignore_index=ignore_index,
    )

    for value in pair.get("input", []):
        _append_token(
            sample,
            int(value),
            tokens=tokens,
            ignore_index=ignore_index,
        )

    _append_token(
        sample,
        tokens.target_id,
        tokens=tokens,
        ignore_index=ignore_index,
        target_id=tokens.target_id,
    )

    for value in pair.get("output", []):
        colour = int(value)
        _append_token(
            sample,
            tokens.mask_id,
            tokens=tokens,
            ignore_index=ignore_index,
            target_id=colour,
            label=colour,
            contributes_to_loss=True,
        )

    _append_token(
        sample,
        tokens.eos_id,
        tokens=tokens,
        ignore_index=ignore_index,
    )

    if len(sample.input_ids) > max_seq_len:
        if drop_long:
            return None
        sample.input_ids = sample.input_ids[:max_seq_len]
        sample.target_ids = sample.target_ids[:max_seq_len]
        sample.labels = sample.labels[:max_seq_len]
        sample.loss_mask = sample.loss_mask[:max_seq_len]

    return sample


def build_query_sample(
    *,
    puzzle_identifier: int,
    episode: Dict[str, object],
    tokens: TokenVocabulary,
    ignore_index: int,
    max_seq_len: int,
    drop_long: bool,
) -> Optional[PuzzleEmbeddingSample]:
    """Construct a sample for the held-out query/solution pair."""

    query_values = episode.get("query") or []
    solution_values = episode.get("solution") or []

    sample = PuzzleEmbeddingSample(
        puzzle_identifier=puzzle_identifier,
        input_ids=[],
        target_ids=[],
        labels=[],
        loss_mask=[],
        sample_type="query",
    )

    _append_token(
        sample,
        tokens.query_id,
        tokens=tokens,
        ignore_index=ignore_index,
    )
    for value in query_values:
        _append_token(
            sample,
            int(value),
            tokens=tokens,
            ignore_index=ignore_index,
        )

    _append_token(
        sample,
        tokens.target_id,
        tokens=tokens,
        ignore_index=ignore_index,
        target_id=tokens.target_id,
    )
    for value in solution_values:
        colour = int(value)
        _append_token(
            sample,
            tokens.mask_id,
            tokens=tokens,
            ignore_index=ignore_index,
            target_id=colour,
            label=colour,
            contributes_to_loss=True,
        )

    _append_token(
        sample,
        tokens.eos_id,
        tokens=tokens,
        ignore_index=ignore_index,
    )

    if len(sample.input_ids) > max_seq_len:
        if drop_long:
            return None
        sample.input_ids = sample.input_ids[:max_seq_len]
        sample.target_ids = sample.target_ids[:max_seq_len]
        sample.labels = sample.labels[:max_seq_len]
        sample.loss_mask = sample.loss_mask[:max_seq_len]

    return sample


class PuzzleEmbeddingIterableDataset(IterableDataset):
    """Iterable dataset that enumerates training pairs across multiple splits."""

    def __init__(
        self,
        *,
        datasets: Sequence[Tuple[str, Iterable[Dict[str, object]]]],
        tokens: TokenVocabulary,
        ignore_index: int,
        max_seq_len: int,
        drop_long: bool,
        identifier_table: PuzzleIdentifierTable,
        shuffle: bool,
        shuffle_buffer: Optional[int],
        seed: int,
        include_query: bool = False,
    ) -> None:
        super().__init__()
        self._datasets = datasets
        self._tokens = tokens
        self._ignore_index = ignore_index
        self._max_seq_len = max_seq_len
        self._drop_long = drop_long
        self._identifier_table = identifier_table
        self._shuffle = shuffle
        self._shuffle_buffer = shuffle_buffer
        self._seed = seed
        self._iteration = 0
        self._include_query = bool(include_query)

    def __iter__(self) -> Iterator[PuzzleEmbeddingSample]:
        if not self._shuffle:
            for split_name, dataset in self._datasets:
                for episode in dataset:
                    episode_id = str(episode.get("id"))
                    puzzle_key = f"{split_name}:{episode_id}"
                    puzzle_identifier = self._identifier_table.get_or_create(puzzle_key)
                    for pair in episode.get("train", []):
                        sample = build_support_sample(
                            puzzle_identifier=puzzle_identifier,
                            pair=pair,
                            tokens=self._tokens,
                            ignore_index=self._ignore_index,
                            max_seq_len=self._max_seq_len,
                            drop_long=self._drop_long,
                        )
                        if sample is not None:
                            yield sample
                    if self._include_query:
                        query_sample = build_query_sample(
                            puzzle_identifier=puzzle_identifier,
                            episode=episode,
                            tokens=self._tokens,
                            ignore_index=self._ignore_index,
                            max_seq_len=self._max_seq_len,
                            drop_long=self._drop_long,
                        )
                        if query_sample is not None:
                            yield query_sample
            return

        # Shuffle with reservoir buffer
        buffer_size = self._shuffle_buffer or 256
        random_seed = self._seed + self._iteration
        self._iteration += 1
        rng = random.Random(random_seed)

        buffer: List[PuzzleEmbeddingSample] = []
        for split_name, dataset in self._datasets:
            for episode in dataset:
                episode_id = str(episode.get("id"))
                puzzle_key = f"{split_name}:{episode_id}"
                puzzle_identifier = self._identifier_table.get_or_create(puzzle_key)
                for pair in episode.get("train", []):
                    sample = build_support_sample(
                        puzzle_identifier=puzzle_identifier,
                        pair=pair,
                        tokens=self._tokens,
                        ignore_index=self._ignore_index,
                        max_seq_len=self._max_seq_len,
                        drop_long=self._drop_long,
                    )
                    if sample is None:
                        continue
                    buffer.append(sample)
                    if len(buffer) >= buffer_size:
                        idx = rng.randrange(len(buffer))
                        yield buffer.pop(idx)
                if self._include_query:
                    query_sample = build_query_sample(
                        puzzle_identifier=puzzle_identifier,
                        episode=episode,
                        tokens=self._tokens,
                        ignore_index=self._ignore_index,
                        max_seq_len=self._max_seq_len,
                        drop_long=self._drop_long,
                    )
                    if query_sample is not None:
                        buffer.append(query_sample)
                        if len(buffer) >= buffer_size:
                            idx = rng.randrange(len(buffer))
                            yield buffer.pop(idx)

        while buffer:
            idx = rng.randrange(len(buffer))
            yield buffer.pop(idx)


def build_puzzle_embedding_collate_fn(
    tokens: TokenVocabulary, ignore_index: int, pad_to_length: Optional[int] = None
):
    """Create a collate function that pads sequences and preserves puzzle ids."""

    pad_token = tokens.pad_id

    def collate(samples: List[PuzzleEmbeddingSample]) -> Dict[str, torch.Tensor]:
        if not samples:
            raise ValueError("Received an empty batch.")
        max_len = max(len(sample.input_ids) for sample in samples)
        if pad_to_length is not None:
            if pad_to_length < max_len:
                raise ValueError(
                    f"pad_to_length={pad_to_length} is smaller than the longest sample ({max_len})."
                )
            max_len = pad_to_length

        batch_size = len(samples)
        inputs = torch.full((batch_size, max_len), pad_token, dtype=torch.long)
        targets = torch.full((batch_size, max_len), pad_token, dtype=torch.long)
        labels = torch.full((batch_size, max_len), ignore_index, dtype=torch.long)
        loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        puzzle_identifiers = torch.zeros(batch_size, dtype=torch.long)
        sample_types = torch.zeros(batch_size, dtype=torch.long)
        type_lookup = {"support": 0, "query": 1}

        for idx, sample in enumerate(samples):
            length = len(sample.input_ids)
            lengths[idx] = length
            puzzle_identifiers[idx] = sample.puzzle_identifier
            inputs[idx, :length] = torch.tensor(sample.input_ids, dtype=torch.long)
            targets[idx, :length] = torch.tensor(sample.target_ids, dtype=torch.long)
            labels[idx, :length] = torch.tensor(sample.labels, dtype=torch.long)
            loss_mask[idx, :length] = torch.tensor(sample.loss_mask, dtype=torch.bool)
            sample_types[idx] = type_lookup.get(sample.sample_type, 0)

        return {
            "inputs": inputs,
            "targets": targets,
            "labels": labels,
            "loss_mask": loss_mask,
            "lengths": lengths,
            "puzzle_identifiers": puzzle_identifiers,
            "sample_types": sample_types,
        }

    return collate
