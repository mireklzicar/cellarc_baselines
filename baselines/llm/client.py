"""OpenAI client utilities for LLM baselines."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from openai import AsyncOpenAI, OpenAI, OpenAIError

LOGGER = logging.getLogger(__name__)


@dataclass
class OpenAIClientConfig:
    """Configuration for OpenAI API calls."""

    model: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: int = 256
    reasoning_effort: Optional[str] = None
    api_style: str = "responses"  # {"responses", "chat"}
    api_key_env: str = "OPENAI_API_KEY"
    base_url: Optional[str] = None
    organization: Optional[str] = None
    request_timeout: Optional[float] = 60.0
    max_retries: int = 2
    retry_backoff: float = 2.0
    stream: bool = False
    use_async: bool = False
    fallback_to_chat_on_empty: bool = True

    def resolve_api_key(self) -> str:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing OpenAI API key. Set the '{self.api_key_env}' environment variable."
            )
        return api_key


class EmptyResponseError(RuntimeError):
    """Raised when the LLM returns no textual content."""


class OpenAIClient:
    """Thin wrapper around the OpenAI SDK with retry logic."""

    def __init__(self, config: OpenAIClientConfig):
        self._config = config
        client_kwargs = {
            "api_key": config.resolve_api_key(),
        }
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        if config.organization:
            client_kwargs["organization"] = config.organization

        self._client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(**client_kwargs) if config.use_async else None
        self._api_style = (config.api_style or "responses").lower()
        if self._api_style not in {"responses", "chat", "chat_completions"}:
            raise ValueError(
                f"Unsupported OpenAI API style '{config.api_style}'. "
                "Use 'responses' or 'chat'."
            )
        if self._api_style == "responses" and not hasattr(self._client, "responses"):
            LOGGER.warning(
                "OpenAI SDK does not expose the 'responses' API; falling back to chat completions."
            )
            self._api_style = "chat"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to OpenAI and return the raw text response."""

        last_error: Optional[Exception] = None
        attempts = self._config.max_retries + 1
        for attempt in range(attempts):
            try:
                if self._api_style == "responses":
                    request_kwargs: dict[str, Any] = self._build_responses_payload(
                        system_prompt,
                        user_prompt,
                    )

                    try:
                        text = self._call_responses_api(request_kwargs)
                    except EmptyResponseError:
                        if self._config.fallback_to_chat_on_empty:
                            LOGGER.warning(
                                "Responses API returned no text. Falling back to chat completions."
                            )
                            text = self._call_chat_api(system_prompt, user_prompt)
                        else:
                            raise
                else:
                    text = self._call_chat_api(system_prompt, user_prompt)

                if not text:
                    raise RuntimeError("Received empty response from OpenAI API.")
                return text
            except OpenAIError as exc:  # pragma: no cover - best effort logging
                last_error = exc
                LOGGER.warning("OpenAI API error: %s", exc)
            except Exception as exc:  # pragma: no cover - logging path
                last_error = exc
                LOGGER.warning("Unexpected OpenAI client error: %s", exc)

            if attempt < attempts - 1:
                backoff = self._config.retry_backoff * (2**attempt)
                LOGGER.info("Retrying OpenAI call in %.1fs (attempt %s/%s).", backoff, attempt + 2, attempts)
                time.sleep(backoff)

        raise RuntimeError("Exhausted retries when calling OpenAI API.") from last_error

    async def generate_async(self, system_prompt: str, user_prompt: str) -> str:
        if not self._async_client:
            raise RuntimeError("Async client requested but use_async=False in config.")

        if self._api_style == "responses":
            if self._config.stream:
                raise RuntimeError("Streaming is not supported in async mode.")
            request_kwargs = self._build_responses_payload(system_prompt, user_prompt)
            try:
                return await self._call_responses_api_async(request_kwargs)
            except EmptyResponseError:
                if self._config.fallback_to_chat_on_empty:
                    LOGGER.warning(
                        "Responses API returned no text (async). Falling back to chat completions."
                    )
                    return await self._call_chat_api_async(system_prompt, user_prompt)
                raise

        return await self._call_chat_api_async(system_prompt, user_prompt)

    def _build_responses_payload(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            }
        )

        payload: dict[str, Any] = {
            "model": self._config.model,
            "input": messages,
            "max_output_tokens": self._config.max_output_tokens,
            "timeout": self._config.request_timeout,
        }
        if self._config.reasoning_effort:
            payload["reasoning"] = {"effort": self._config.reasoning_effort}
        return payload

    def _call_responses_api(self, request_kwargs: dict[str, Any]) -> str:
        if self._config.stream:
            text = _stream_response_text(self._client, request_kwargs)
        else:
            response = self._client.responses.create(**request_kwargs)
            text = _extract_response_text(response)

        if not text:
            status = getattr(response, "status", None)
            error = getattr(response, "error", None)
            details: list[str] = []
            if status:
                details.append(f"status={status}")
            if error:
                details.append(f"error={error}")
            detail_msg = f" ({'; '.join(details)})" if details else ""
            raise EmptyResponseError(f"Received empty response from OpenAI API{detail_msg}.")
        return text

    async def _call_responses_api_async(self, request_kwargs: dict[str, Any]) -> str:
        response = await self._async_client.responses.create(**request_kwargs)
        text = _extract_response_text(response)
        if not text:
            status = getattr(response, "status", None)
            error = getattr(response, "error", None)
            details: list[str] = []
            if status:
                details.append(f"status={status}")
            if error:
                details.append(f"error={error}")
            detail_msg = f" ({'; '.join(details)})" if details else ""
            raise EmptyResponseError(f"Received empty response from OpenAI API{detail_msg}.")
        return text

    def _call_chat_api(self, system_prompt: str, user_prompt: str) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "max_completion_tokens": self._config.max_output_tokens,
            "timeout": self._config.request_timeout,
        }
        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature
        if self._config.top_p is not None:
            kwargs["top_p"] = self._config.top_p

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        text = (getattr(message, "content", None) or "").strip()
        if not text:
            refusal = (getattr(message, "refusal", None) or "").strip()
            finish_reason = getattr(choice, "finish_reason", None)
            details: list[str] = []
            if finish_reason:
                details.append(f"finish_reason={finish_reason}")
            if refusal:
                details.append(f"refusal={refusal}")
            detail_msg = f" ({'; '.join(details)})" if details else ""
            raise EmptyResponseError(f"Received empty response from OpenAI API{detail_msg}.")
        return text

    async def _call_chat_api_async(self, system_prompt: str, user_prompt: str) -> str:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "max_completion_tokens": self._config.max_output_tokens,
            "timeout": self._config.request_timeout,
        }
        if self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature
        if self._config.top_p is not None:
            kwargs["top_p"] = self._config.top_p

        response = await self._async_client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        text = (getattr(message, "content", None) or "").strip()
        if not text:
            refusal = (getattr(message, "refusal", None) or "").strip()
            finish_reason = getattr(choice, "finish_reason", None)
            details: list[str] = []
            if finish_reason:
                details.append(f"finish_reason={finish_reason}")
            if refusal:
                details.append(f"refusal={refusal}")
            detail_msg = f" ({'; '.join(details)})" if details else ""
            raise EmptyResponseError(f"Received empty response from OpenAI API{detail_msg}.")
        return text


def _extract_response_text(response: Any) -> str:
    """Extract concatenated text from a responses.create result."""

    chunks: list[str] = []
    output = getattr(response, "output", None)
    if output:
        for item in output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for block in content:
                text = getattr(block, "text", None)
                if text:
                    chunks.append(str(text))

    if not chunks:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, list):
            chunks.extend(str(part) for part in output_text if part)
        elif output_text:
            chunks.append(str(output_text))

    aggregated = "\n".join(chunk.strip() for chunk in chunks if chunk).strip()
    return aggregated


def _stream_response_text(client: OpenAI, request_kwargs: dict[str, Any]) -> str:
    from contextlib import suppress

    chunks: list[str] = []
    with client.responses.stream(**request_kwargs) as stream:
        for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", None)
                if delta:
                    chunks.append(str(delta))
            elif event_type == "response.output_text.done":
                break
        final_response = None
        with suppress(Exception):
            final_response = stream.get_final_response()

    if chunks:
        return "".join(chunks).strip()

    if final_response is not None:
        return _extract_response_text(final_response)

    raise RuntimeError("Streaming response produced no text.")
