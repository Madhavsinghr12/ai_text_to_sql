"""The SQL guard. The model's output is untrusted input, so it gets parsed before
it ever meets a database.

Four checks, cheapest first:

1. it must parse as exactly one statement;
2. that statement must be a read (SELECT / UNION of SELECTs);
3. every referenced table must be on the allowlist;
4. a row cap is applied so a stray ``SELECT *`` cannot stream a million rows.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

DIALECT = "sqlite"
DEFAULT_MAX_ROWS = 200

_READ_NODES = (exp.Select, exp.Union, exp.Intersect, exp.Except)

# Referenced defensively through getattr: sqlglot reshuffles some node classes
# between minor versions, and an AttributeError at import time would take the
# whole package down for a node we do not actually need.
_FORBIDDEN_NODE_NAMES = (
    "Insert",
    "Update",
    "Delete",
    "Create",
    "Drop",
    "Alter",
    "TruncateTable",
    "Merge",
    "Attach",
    "Detach",
    "Pragma",
    "Transaction",
    "Commit",
    "Rollback",
    "Grant",
    "Use",
    "Set",
    "Copy",
    "Call",
    "LoadData",
    "Describe",
    "Command",
)

# Functions that reach outside the data: file I/O, extension loading, process
# execution. ``load_extension`` is how SQLite is normally used to escape read-only.
_FORBIDDEN_FUNCTIONS = {
    "load_extension",
    "readfile",
    "writefile",
    "edit",
    "eval",
    "system",
    "shell",
    "fts3_tokenizer",
    "sqlite_compileoption_get",
}


class SQLRejected(ValueError):
    """Raised when generated SQL is unsafe or malformed. The message is fed back
    to the model verbatim, so it is phrased as a correction prompt."""


def validate(
    sql: str, allowed_tables: set[str] | list[str], max_rows: int = DEFAULT_MAX_ROWS
) -> str:
    """Return executable SQL, or raise ``SQLRejected`` with a fixable message."""
    text = sql.strip().rstrip(";").strip()
    if not text:
        raise SQLRejected("The response contained no SQL. Return one SELECT statement.")

    statements = _parse(text)
    if len(statements) != 1:
        raise SQLRejected(
            f"Exactly one statement is allowed, but the response had {len(statements)}. "
            "Return a single SELECT statement."
        )
    statement = statements[0]

    if not isinstance(statement, _READ_NODES):
        raise SQLRejected(
            "Only read-only SELECT queries are allowed. "
            f"This is a {statement.key.upper()} statement."
        )

    for name in _FORBIDDEN_NODE_NAMES:
        node = getattr(exp, name, None)
        if node is not None and statement.find(node) is not None:
            raise SQLRejected(f"{name.upper()} is not allowed. Return a single SELECT statement.")

    _check_tables(statement, set(allowed_tables))
    _check_functions(statement)
    return _cap_rows(statement, max_rows)


def _parse(text: str) -> list[exp.Expression]:
    try:
        return [s for s in sqlglot.parse(text, read=DIALECT) if s is not None]
    except ParseError as exc:
        detail = str(exc).splitlines()[0]
        raise SQLRejected(
            f"Could not parse that SQL ({detail}). Return one valid SQLite SELECT."
        ) from exc


def _check_tables(statement: exp.Expression, allowed: set[str]) -> None:
    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    referenced = {table.name.lower() for table in statement.find_all(exp.Table) if table.name}
    unknown = referenced - cte_names - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise SQLRejected(
            f"Unknown table(s): {names}. Use only these tables: {', '.join(sorted(allowed))}."
        )


def _check_functions(statement: exp.Expression) -> None:
    for func in statement.find_all(exp.Func):
        name = func.name.lower() if isinstance(func, exp.Anonymous) else func.sql_name().lower()
        if name in _FORBIDDEN_FUNCTIONS:
            raise SQLRejected(f"The function {name}() is not allowed.")


def _cap_rows(statement: exp.Expression, max_rows: int) -> str:
    limit = statement.args.get("limit")
    current = None
    if isinstance(limit, exp.Limit) and isinstance(limit.expression, exp.Literal):
        if limit.expression.is_int:
            current = int(limit.expression.name)
    if current is None or current > max_rows:
        # Replacing the limit rather than wrapping in a subquery keeps the
        # inner ORDER BY authoritative for top-N questions.
        statement = statement.limit(max_rows)
    return statement.sql(dialect=DIALECT, pretty=True)
