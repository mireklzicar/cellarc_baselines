"""Prompt construction helpers for LLM baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_SYSTEM_MESSAGE = (
    "You are a careful assistant that solves Cellular ARC tasks. "
    "Infer the rule from the provided 1D training sequences and apply it to the test input. "
    "Think through the problem if needed, but only return the final output sequence."
)

DEFAULT_RESPONSE_HINT = (
    "Return only the predicted output sequence as space-separated integers (e.g., `3 1 2 0 1`)."
)


@dataclass
class PromptSpec:
    """Container holding the system and user prompts."""

    system: str
    user: str


@dataclass
class PromptBuilderConfig:
    """Configuration values that influence prompt construction."""

    instructions: str
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    response_format_hint: str = DEFAULT_RESPONSE_HINT
    max_train_examples: int | None = None
    include_episode_id: bool = False


class PromptBuilder:
    """Creates textual prompts for Cellular ARC episodes."""

    def __init__(self, config: PromptBuilderConfig):
        self._config = config
        self._instructions = (config.instructions or "").strip()
        if not self._instructions:
            raise ValueError("Prompt instructions must be a non-empty string.")
        if config.system_message is None:
            self._system_message = DEFAULT_SYSTEM_MESSAGE
        else:
            self._system_message = config.system_message.strip()
        self._response_hint = (config.response_format_hint or DEFAULT_RESPONSE_HINT).strip()

    def build_prompt(self, episode: Mapping[str, Any]) -> PromptSpec:
        """Convert an episode into a prompt compatible with our LLM caller."""

        train_pairs = list(episode.get("train", []))
        if self._config.max_train_examples is not None:
            train_pairs = train_pairs[: self._config.max_train_examples]

        query = episode.get("query")
        if query is None:
            raise ValueError("Episode is missing a 'query' field required for inference.")

        lines: list[str] = [self._instructions, ""]
        if self._config.include_episode_id:
            episode_id = episode.get("id")
            if episode_id:
                lines.append(f"Episode ID: {episode_id}")
                lines.append("")

        if train_pairs:
            for index, pair in enumerate(train_pairs, start=1):
                input_seq = _format_sequence(pair.get("input", []))
                output_seq = _format_sequence(pair.get("output", []))
                lines.append(f"Example {index}:")
                lines.append("")
                lines.append("Input:")
                lines.append(input_seq)
                lines.append("Output:")
                lines.append(output_seq)
                lines.append("")
        else:
            lines.append("No examples were provided. Rely on your reasoning ability.")
            lines.append("")

        lines.append(
            "Below is the test input sequence. Predict the corresponding output sequence."
        )
        lines.append(self._response_hint)
        lines.append("")
        lines.append("Input:")
        lines.append(_format_sequence(query))

        user_prompt = "\n".join(lines).strip()
        return PromptSpec(system=self._system_message, user=user_prompt)


def _format_sequence(seq: Sequence[int] | Iterable[int]) -> str:
    return " ".join(str(int(token)) for token in seq)
