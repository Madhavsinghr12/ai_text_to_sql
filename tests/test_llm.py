from __future__ import annotations

import openai
import pytest

from text2sql import cli
from text2sql.llm import (
    DEFAULT_MODEL,
    PUBLIC_BASE_URL,
    PUBLIC_MODEL,
    ModelUnavailable,
    OpenAIModel,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("T2SQL_API_KEY", "OPENAI_API_KEY", "T2SQL_BASE_URL", "T2SQL_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)


def test_no_key_and_no_endpoint_uses_the_public_endpoint() -> None:
    model = OpenAIModel()
    assert model.public
    assert model.base_url == PUBLIC_BASE_URL
    assert model.model == PUBLIC_MODEL


def test_a_key_keeps_the_configured_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("T2SQL_API_KEY", "sk-test")
    model = OpenAIModel()
    assert not model.public
    assert model.base_url is None
    assert model.model == DEFAULT_MODEL


def test_a_local_endpoint_does_not_need_a_key() -> None:
    model = OpenAIModel(base_url="http://localhost:1234/v1")
    assert not model.public
    assert model.base_url == "http://localhost:1234/v1"


class _Refusing:
    def create(self, **kwargs):
        raise openai.APIConnectionError(request=None)


def test_endpoint_errors_become_model_unavailable() -> None:
    model = OpenAIModel()
    model._client.chat.completions = _Refusing()
    with pytest.raises(ModelUnavailable, match="refused the request"):
        model.complete([{"role": "user", "content": "hi"}])


class _RateLimited:
    public = False

    def complete(self, messages):
        raise ModelUnavailable("The model endpoint refused the request: rate limited")


def test_cli_reports_a_refused_request_without_a_traceback(
    demo_db, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "OpenAIModel", lambda **kwargs: _RateLimited())
    code = cli.main(["How many orders were cancelled?", "--db", str(demo_db)])
    err = capsys.readouterr().err
    assert code == 3
    assert "rate limited" in err
    assert "Traceback" not in err
