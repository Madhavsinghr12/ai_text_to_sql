"""Model transport.

One client, any OpenAI-compatible endpoint: OpenAI, Groq, OpenRouter, Together,
vLLM, LM Studio. Point ``T2SQL_BASE_URL`` at the endpoint and set a key. With no
key and no endpoint, the free keyless Pollinations endpoint is used.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

DEFAULT_MODEL = "gpt-4o-mini"
PUBLIC_BASE_URL = "https://text.pollinations.ai/openai"
PUBLIC_MODEL = "openai"


class ModelUnavailable(RuntimeError):
    """The endpoint refused or failed the request: rate limit, quota, auth, network."""


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
        base_url = base_url or os.getenv("T2SQL_BASE_URL") or None
        self.public = not key and not base_url
        if self.public:
            base_url = PUBLIC_BASE_URL

        self.model = (
            model or os.getenv("T2SQL_MODEL") or (PUBLIC_MODEL if self.public else DEFAULT_MODEL)
        )
        self.base_url = base_url
        self.temperature = temperature
        # Keyless endpoints (Pollinations, LM Studio, vLLM) ignore the key; the client needs one.
        self._client = OpenAI(api_key=key or "not-needed", base_url=base_url, timeout=timeout)

    def complete(self, messages: list[dict[str, str]]) -> str:
        from openai import APIError

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
        except APIError as exc:
            raise ModelUnavailable(
                f"The model endpoint refused the request: {exc.message}"
            ) from exc
        return (response.choices[0].message.content or "").strip()

    def __repr__(self) -> str:
        return f"OpenAIModel(model={self.model!r})"
