"""The eval set is data, so it gets tested like data: every reference query must
run, and the comparison must actually be capable of rejecting a wrong answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from run_eval import matches  # noqa: E402

from text2sql.guard import validate  # noqa: E402
from text2sql.schema import table_names  # noqa: E402

QUESTIONS = yaml.safe_load((ROOT / "eval" / "questions.yaml").read_text(encoding="utf-8"))


def test_eval_set_has_forty_questions() -> None:
    assert len(QUESTIONS) == 40


def test_ids_are_unique_and_every_question_has_reference_sql() -> None:
    ids = [q["id"] for q in QUESTIONS]
    assert len(set(ids)) == len(ids)
    assert all(q.get("reference_sql") for q in QUESTIONS)
    assert {q["difficulty"] for q in QUESTIONS} == {"basic", "intermediate", "advanced"}


@pytest.mark.parametrize("question", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_reference_sql_passes_the_guard(question: dict, demo_db) -> None:
    """A reference query the guard rejects would make the expectation unreachable."""
    assert validate(question["reference_sql"], set(table_names(demo_db)))


def test_unordered_comparison_ignores_row_order() -> None:
    expected = [("US", 10), ("UK", 5)]
    actual = [("UK", 5), ("US", 10)]
    assert matches(expected, actual, order_matters=False, decimals=2)


def test_ordered_comparison_rejects_wrong_ranking() -> None:
    expected = [("US", 10), ("UK", 5)]
    actual = [("UK", 5), ("US", 10)]
    assert not matches(expected, actual, order_matters=True, decimals=2)


def test_comparison_tolerates_rounding_and_int_float_equivalence() -> None:
    assert matches([(1234.5678,)], [(1234.57,)], order_matters=False, decimals=2)
    assert matches([(1234,)], [(1234.0,)], order_matters=False, decimals=2)


def test_comparison_rejects_wrong_values_row_counts_and_type_changes() -> None:
    assert not matches([(10,)], [(11,)], order_matters=False, decimals=2)
    assert not matches([(10,), (20,)], [(10,)], order_matters=False, decimals=2)
    assert not matches([("web", 1)], [(1, "web")], order_matters=False, decimals=2)
