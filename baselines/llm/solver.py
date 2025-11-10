"""LLM solver abstractions."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Mapping

from .client import OpenAIClient
from .prompting import PromptBuilder

LOGGER = logging.getLogger(__name__)

_INT_PATTERN = re.compile(r"-?\d+")


@dataclass
class PredictionParserConfig:
    """Controls how raw LLM responses are parsed."""

    clip_to_expected: bool = True
    allow_partial: bool = True


class PredictionParser:
    """Extract integer sequences from raw model text."""

    def __init__(self, config: PredictionParserConfig):
        self._config = config

    def parse(self, text: str, expected_length: int | None = None) -> List[int]:
        matches = [int(match) for match in _INT_PATTERN.findall(text or "")]
        if expected_length is not None and self._config.clip_to_expected:
            matches = matches[:expected_length]

        if expected_length is not None and not matches and not self._config.allow_partial:
            raise ValueError("LLM response did not contain any integers.")

        if (
            expected_length is not None
            and len(matches) < expected_length
            and not self._config.allow_partial
        ):
            raise ValueError(
                f"Expected {expected_length} symbols but parsed only {len(matches)}."
            )

        return matches


@dataclass
class LLMPrediction:
    """Package the structured prediction along with the raw text."""

    tokens: List[int]
    raw_text: str
    prompt_text: str


class LLMSolver:
    """Generate predictions for Cellular ARC episodes via an LLM."""

    def __init__(
        self,
        client: OpenAIClient,
        prompt_builder: PromptBuilder,
        parser: PredictionParser,
    ):
        self._client = client
        self._prompt_builder = prompt_builder
        self._parser = parser

    def predict(self, episode: Mapping[str, object], expected_length: int | None) -> LLMPrediction:
        prompt = self._prompt_builder.build_prompt(episode)
        raw_text = self._client.generate(prompt.system, prompt.user)
        try:
            tokens = self._parser.parse(raw_text, expected_length=expected_length)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to parse LLM response: %s", exc)
            tokens = []
        return LLMPrediction(tokens=tokens, raw_text=raw_text, prompt_text=prompt.user)

    async def predict_async(
        self,
        episode: Mapping[str, object],
        expected_length: int | None,
    ) -> LLMPrediction:
        prompt = self._prompt_builder.build_prompt(episode)
        raw_text = await self._client.generate_async(prompt.system, prompt.user)
        try:
            tokens = self._parser.parse(raw_text, expected_length=expected_length)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to parse LLM response: %s", exc)
            tokens = []
        return LLMPrediction(tokens=tokens, raw_text=raw_text, prompt_text=prompt.user)
