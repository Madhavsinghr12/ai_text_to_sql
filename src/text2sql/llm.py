"""Model transport.

One client, any OpenAI-compatible endpoint: OpenAI, Groq, OpenRouter, Together,
vLLM, LM Studio. Point ``T2SQL_BASE_URL`` at the endpoint and set a key.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

DEFAULT_MODEL = "gpt-4o-mini"


@runtime_checkable
class Model(Protocol):
    """Anything that turns chat messages into text. Tests use a scripted stand-in."""

    def complete(self, messages: list[dict[str, str]]) -> str: ...


class OpenAIModel:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - install-time problem
            raise RuntimeError("The openai package is required: pip install openai") from exc

        try:  # .env is a convenience, never a requirement
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:  # pragma: no cover
            pass

        key = api_key or os.getenv("T2SQL_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "No API key found. Set T2SQL_API_KEY (or OPENAI_API_KEY) in your "
                "environment or .env file. See .env.example."
            )

        self.model = model or os.getenv("T2SQL_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self._client = OpenAI(
            api_key=key,
            base_url=base_url or os.getenv("T2SQL_BASE_URL") or None,
            timeout=timeout,
        )

    def complete(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def __repr__(self) -> str:
        return f"OpenAIModel(model={self.model!r})"
