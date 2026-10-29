"""The loop: grounding, retry-on-error, refusing to retry forever, and never
letting a bad generation reach the data."""

from __future__ import annotations

import sqlite3

import pytest

from text2sql.agent import TextToSQLAgent, extract_sql, parse_clarification
from text2sql.testing import ScriptedModel

GOOD_SQL = (
    "SELECT country, COUNT(*) AS customers FROM customers GROUP BY country ORDER BY customers DESC"
)
GOOD_RESPONSE = f"```sql\n{GOOD_SQL}\n```"


def make_agent(model, demo_db, **kwargs) -> TextToSQLAgent:
    kwargs.setdefault("summarise", False)
    return TextToSQLAgent(model, demo_db, **kwargs)


class SummaryFailsModel:
    """Answers the SQL call, then blows up on the summary call."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        if self.calls == 1:
            return GOOD_RESPONSE
        raise RuntimeError("upstream 500")


def test_prompt_is_grounded_in_the_real_schema(demo_db) -> None:
    agent = make_agent(ScriptedModel([]), demo_db)
    prompt = agent.system_prompt
    assert "TABLE orders" in prompt
    assert "fk -> customers.id" in prompt
    assert "order_items.unit_price" in prompt  # the revenue convention
    assert agent.today.isoformat() in prompt
    assert "CLARIFY:" in prompt


def test_happy_path_returns_rows_and_sql(demo_db) -> None:
    result = make_agent(ScriptedModel([GOOD_RESPONSE]), demo_db).ask("customers per country?")
    assert result.ok
    assert len(result.rows) == 8
    assert result.columns == ["country", "customers"]
    assert "LIMIT" in result.sql
    assert result.answer


def test_answer_comes_from_the_summary_call(demo_db) -> None:
    model = ScriptedModel([GOOD_RESPONSE, "The United States has the most customers."])
    result = make_agent(model, demo_db, summarise=True).ask("customers per country?")
    assert result.answer == "The United States has the most customers."
    assert len(model.calls) == 2


def test_broken_summary_does_not_lose_the_rows(demo_db) -> None:
    result = make_agent(SummaryFailsModel(), demo_db, summarise=True).ask("customers per country?")
    assert result.ok
    assert len(result.rows) == 8
    assert result.answer  # deterministic fallback


def test_error_is_fed_back_and_retried(demo_db) -> None:
    model = ScriptedModel(["```sql\nSELECT * FROM sales\n```", GOOD_RESPONSE])
    result = make_agent(model, demo_db).ask("customers per country?")
    assert result.ok
    assert len(result.attempts) == 2
    assert "Unknown table" in (result.attempts[0].error or "")
    assert "Unknown table" in model.calls[1][-1]["content"]


def test_gives_up_after_the_configured_corrections(demo_db) -> None:
    model = ScriptedModel(["not sql"] * 3)
    result = make_agent(model, demo_db, corrections=2).ask("anything")
    assert not result.ok
    assert len(result.attempts) == 3
    assert len(model.calls) == 3
    assert "Gave up" in (result.error or "")


def test_zero_corrections_means_one_attempt(demo_db) -> None:
    model = ScriptedModel(["not sql"])
    result = make_agent(model, demo_db, corrections=0).ask("anything")
    assert not result.ok
    assert len(model.calls) == 1


def test_unsafe_sql_never_reaches_the_database(demo_db) -> None:
    model = ScriptedModel(["```sql\nDROP TABLE orders\n```"] * 3)
    result = make_agent(model, demo_db).ask("delete all orders")
    assert not result.ok
    with sqlite3.connect(demo_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 5200


def test_ambiguous_question_gets_a_clarification_instead_of_a_query(demo_db) -> None:
    model = ScriptedModel(["CLARIFY: Should revenue be gross or net of returns?"])
    result = make_agent(model, demo_db).ask("top products by revenue")
    assert result.clarification == "Should revenue be gross or net of returns?"
    assert result.sql is None
    assert result.rows == []
    assert len(model.calls) == 1


def test_max_rows_is_enforced_end_to_end(demo_db) -> None:
    result = make_agent(
        ScriptedModel(["```sql\nSELECT * FROM customers\n```"]), demo_db, max_rows=3
    ).ask("all customers")
    assert len(result.rows) == 3


@pytest.mark.parametrize(
    "raw",
    [
        "```sql\nSELECT 1\n```",
        "```\nSELECT 1\n```",
        "SQL: SELECT 1",
        "SELECT 1",
        "Here you go:\n```sql\nSELECT 1\n```\nThis counts rows.",
    ],
)
def test_extract_sql_handles_common_response_shapes(raw: str) -> None:
    assert extract_sql(raw) == "SELECT 1"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CLARIFY: Which month?", "Which month?"),
        ("clarify: which month?", "which month?"),
        ("CLARIFY:   Which\nmonth?  ", "Which month?"),
        ("SELECT 1", None),
        ("", None),
    ],
)
def test_parse_clarification(raw: str, expected: str | None) -> None:
    assert parse_clarification(raw) == expected
