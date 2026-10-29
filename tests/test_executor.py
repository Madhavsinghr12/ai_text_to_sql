"""Execution tests. These deliberately bypass the guard, because the point is to
prove the connection is read-only even when nobody has checked the SQL."""

from __future__ import annotations

import pytest

from text2sql.executor import QueryError, QueryTimeout, execute

ENDLESS = (
    "WITH RECURSIVE counter(x) AS (SELECT 1 UNION ALL "
    "SELECT x + 1 FROM counter WHERE x < 100000000) SELECT COUNT(*) FROM counter"
)


def test_returns_columns_and_rows(demo_db) -> None:
    result = execute(demo_db, "SELECT id, name FROM categories ORDER BY id LIMIT 3")
    assert result.columns == ["id", "name"]
    assert len(result.rows) == 3
    assert result.rows[0][0] == 1


def test_row_cap_truncates_and_reports(demo_db) -> None:
    result = execute(demo_db, "SELECT * FROM orders", max_rows=5)
    assert len(result.rows) == 5
    assert result.truncated is True


def test_does_not_report_truncation_when_rows_fit(demo_db) -> None:
    result = execute(demo_db, "SELECT * FROM categories", max_rows=5000)
    assert result.truncated is False


def test_write_is_refused_by_the_connection(demo_db) -> None:
    with pytest.raises(QueryError) as excinfo:
        execute(demo_db, "DELETE FROM orders")
    assert "readonly" in str(excinfo.value).lower()


def test_runaway_query_is_cancelled(demo_db) -> None:
    with pytest.raises(QueryTimeout):
        execute(demo_db, ENDLESS, timeout_s=0.05)


def test_missing_database_is_a_query_error(tmp_path) -> None:
    with pytest.raises(QueryError):
        execute(tmp_path / "nope.db", "SELECT 1")
