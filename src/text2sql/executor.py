"""Query execution against a read-only connection, with a wall-clock cap.

Three independent layers keep this read-only:
  * the URI is ``mode=ro``;
  * ``PRAGMA query_only`` is set on the connection;
  * the guard has already rejected anything that is not a SELECT.

Any one of them would be enough. They are stacked because generated SQL is the
most exposed input surface in this whole program.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_ROWS = 200
DEFAULT_TIMEOUT_S = 5.0

_PROGRESS_INTERVAL = 20_000  # VM instructions between timeout checks


class QueryError(RuntimeError):
    """The database rejected the query. Message is safe to show the model."""


class QueryTimeout(QueryError):
    """The query ran longer than the allowed time."""


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool
    elapsed_ms: float


def execute(
    db_path: str | Path,
    sql: str,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> QueryResult:
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    started = time.perf_counter()
    conn: sqlite3.Connection | None = None

    try:
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=timeout_s, isolation_level=None)
            conn.execute("PRAGMA query_only = ON")
            deadline = time.monotonic() + timeout_s
            conn.set_progress_handler(_watchdog(deadline), _PROGRESS_INTERVAL)

            cursor = conn.execute(sql)
            columns = [description[0] for description in cursor.description or ()]
            rows = cursor.fetchmany(max_rows + 1)
        except sqlite3.OperationalError as exc:
            raise _translate(exc, timeout_s) from exc
        except sqlite3.Error as exc:
            raise QueryError(str(exc)) from exc
    finally:
        if conn is not None:
            conn.close()

    truncated = len(rows) > max_rows
    return QueryResult(
        columns=columns,
        rows=[tuple(row) for row in rows[:max_rows]],
        truncated=truncated,
        elapsed_ms=(time.perf_counter() - started) * 1000,
    )


def _watchdog(deadline: float):
    def check() -> int:
        return int(time.monotonic() > deadline)

    return check


def _translate(exc: sqlite3.OperationalError, timeout_s: float) -> QueryError:
    message = str(exc)
    if "interrupted" in message.lower():
        return QueryTimeout(f"The query took longer than {timeout_s:g}s and was cancelled.")
    return QueryError(message)
