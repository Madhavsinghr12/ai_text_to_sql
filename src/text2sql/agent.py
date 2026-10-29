"""The agent loop.

    question -> prompt (schema + examples) -> SQL -> guard -> execute
                 ^                                          |
                 +------------- error fed back ------------+

Up to ``corrections`` retries, then a plain-English answer built from the rows.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from text2sql.executor import QueryError, execute
from text2sql.guard import DEFAULT_MAX_ROWS, SQLRejected, validate
from text2sql.llm import Model
from text2sql.schema import render_schema, table_names

CLARIFY_PREFIX = "CLARIFY:"
DEFAULT_CORRECTIONS = 2
PREVIEW_ROWS = 20

# In a production warehouse these conventions live in a semantic layer (dbt docs,
# a metrics catalogue). Here they are short enough to inline.
DOMAIN_NOTES = """\
- An order's revenue is SUM(order_items.quantity * order_items.unit_price).
- Only orders with status = 'completed' count as realised revenue. Other status
  values are 'cancelled', 'refunded' and 'pending'.
- order_items.unit_price is the price actually charged, already discounted.
- returns row reference an order_item, so net revenue would subtract returned items.
- products.active = 1 means still sold, 0 means discontinued.
- orders.channel is one of 'web', 'mobile', 'store'."""

FEW_SHOT: list[tuple[str, str]] = [
    (
        "How many orders were placed in 2025?",
        "SELECT COUNT(*) AS order_count\n"
        "FROM orders\n"
        "WHERE order_date >= '2025-01-01' AND order_date < '2026-01-01'",
    ),
    (
        "Which 5 products brought in the most revenue in September 2026?",
        "SELECT p.name,\n"
        "       ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue\n"
        "FROM order_items AS oi\n"
        "JOIN orders AS o   ON o.id = oi.order_id\n"
        "JOIN products AS p ON p.id = oi.product_id\n"
        "WHERE o.status = 'completed'\n"
        "  AND o.order_date >= '2026-09-01' AND o.order_date < '2026-10-01'\n"
        "GROUP BY p.id, p.name\n"
        "ORDER BY revenue DESC\n"
        "LIMIT 5",
    ),
    (
        "What is the average order value for completed orders?",
        "SELECT ROUND(AVG(order_total), 2) AS avg_order_value\n"
        "FROM (\n"
        "    SELECT o.id, SUM(oi.quantity * oi.unit_price) AS order_total\n"
        "    FROM orders AS o\n"
        "    JOIN order_items AS oi ON oi.order_id = o.id\n"
        "    WHERE o.status = 'completed'\n"
        "    GROUP BY o.id\n"
        ")",
    ),
    (
        "How many customers are there per country?",
        "SELECT country, COUNT(*) AS customers\n"
        "FROM customers\n"
        "GROUP BY country\n"
        "ORDER BY customers DESC",
    ),
    (
        "Which return reasons are most common?",
        "SELECT reason, COUNT(*) AS return_count\n"
        "FROM returns\n"
        "GROUP BY reason\n"
        "ORDER BY return_count DESC",
    ),
]


@dataclass(frozen=True)
class Attempt:
    """One generation. ``error`` is None when the query ran."""

    sql: str | None
    error: str | None = None


@dataclass
class Result:
    question: str
    sql: str | None = None
    answer: str | None = None
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False
    clarification: str | None = None
    error: str | None = None
    attempts: list[Attempt] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and self.clarification is None


def build_system_prompt(schema: str, today: date, max_rows: int, tables: list[str]) -> str:
    return f"""You are a senior analytics engineer who writes SQLite for a retail analytics team.

TASK
Translate the manager's question into exactly ONE read-only SQLite SELECT query.

DATABASE
{schema}

SEMANTICS
{DOMAIN_NOTES}
- Dates are ISO text, 'YYYY-MM-DD'. Today is {today.isoformat()}.

RULES
1. One SELECT statement only. No INSERT/UPDATE/DELETE/DDL, no PRAGMA, no ATTACH.
2. Use only the tables above ({", ".join(tables)}) and only columns that exist.
3. Bound date ranges explicitly: >= start AND < the day after the end.
4. If the question implies a ranking, add ORDER BY and a LIMIT.
5. Round money to 2 decimals and alias every computed column.
6. At most {max_rows} rows are returned, so aggregate rather than dumping raw rows.

AMBIGUITY
If answering requires a decision the question does not make -- "revenue" gross vs
net of returns, "last month" as a calendar month vs the last 30 days, product vs
category breakdown -- do not guess. Reply with exactly one line:
CLARIFY: <one-sentence question>

OUTPUT
Otherwise reply with a single ```sql fenced block and no other text."""


class TextToSQLAgent:
    def __init__(
        self,
        model: Model,
        db_path: str | Path,
        *,
        max_rows: int = DEFAULT_MAX_ROWS,
        corrections: int = DEFAULT_CORRECTIONS,
        timeout_s: float = 5.0,
        summarise: bool = True,
        today: date | None = None,
    ) -> None:
        self.model = model
        self.db_path = Path(db_path)
        self.max_rows = max_rows
        self.corrections = max(0, corrections)
        self.timeout_s = timeout_s
        self.summarise = summarise
        self.today = today or date.today()
        self.tables = table_names(self.db_path)
        self.schema = render_schema(self.db_path)
        self.system_prompt = build_system_prompt(self.schema, self.today, max_rows, self.tables)

    def ask(self, question: str) -> Result:
        started = time.perf_counter()
        messages = self._conversation(question)
        attempts: list[Attempt] = []

        def elapsed_ms() -> float:
            return (time.perf_counter() - started) * 1000

        for round_index in range(self.corrections + 1):
            raw = self.model.complete(messages)

            clarification = parse_clarification(raw)
            if clarification:
                return Result(
                    question=question,
                    clarification=clarification,
                    attempts=attempts,
                    elapsed_ms=elapsed_ms(),
                )

            sql = extract_sql(raw)
            try:
                guarded = validate(sql, self.tables, self.max_rows)
                query = execute(self.db_path, guarded, self.max_rows, self.timeout_s)
            except (SQLRejected, QueryError) as exc:
                attempts.append(Attempt(sql, str(exc)))
                if round_index == self.corrections:
                    return Result(
                        question=question,
                        error=f"Gave up after {round_index + 1} attempt(s). Last error: {exc}",
                        attempts=attempts,
                        elapsed_ms=elapsed_ms(),
                    )
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"That query failed: {exc}\n"
                            "Fix it and return the corrected SQL in a ```sql block."
                        ),
                    },
                ]
                continue

            attempts.append(Attempt(sql))
            answer = (
                self._summarise(question, guarded, query) if self.summarise else None
            ) or fallback_answer(query)
            return Result(
                question=question,
                sql=guarded,
                answer=answer,
                columns=query.columns,
                rows=query.rows,
                truncated=query.truncated,
                attempts=attempts,
                elapsed_ms=elapsed_ms(),
            )

        raise AssertionError("unreachable")  # pragma: no cover

    def _conversation(self, question: str) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for example_question, example_sql in FEW_SHOT:
            messages.append({"role": "user", "content": example_question})
            messages.append({"role": "assistant", "content": f"```sql\n{example_sql}\n```"})
        messages.append({"role": "user", "content": question})
        return messages

    def _summarise(self, question: str, sql: str, result) -> str | None:
        """Second call: rows -> one or two sentences. Numbers must come from the rows."""
        preview = render_preview(result.columns, result.rows, PREVIEW_ROWS)
        instructions = (
            "You summarise query results for a busy manager. Answer in one or two "
            "sentences, with the key number first. Use only values present in the "
            "data. Never invent numbers, never explain the SQL. If there are no "
            "rows, say that no rows matched."
        )
        payload = (
            f"Question: {question}\n\n"
            f"SQL executed:\n{sql}\n\n"
            f"Columns: {', '.join(result.columns)}\n"
            f"Rows ({len(result.rows)} returned{', truncated' if result.truncated else ''}):\n"
            f"{preview}"
        )
        try:
            text = self.model.complete(
                [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": payload},
                ]
            ).strip()
        except Exception:  # a broken summary must not lose the rows
            return None
        return text or None


def extract_sql(text: str) -> str:
    """Pull SQL out of a fenced block, tolerating a ``SQL:`` label or stray prose."""
    fenced = re.search(r"```(?:sql)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    candidate = re.sub(r"^\s*(sql|query)\s*:\s*", "", candidate, flags=re.IGNORECASE)
    return candidate.strip()


def parse_clarification(text: str) -> str | None:
    match = re.match(r"\s*CLARIFY\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return " ".join(match.group(1).split())


def render_preview(columns: list[str], rows: list[tuple], limit: int = PREVIEW_ROWS) -> str:
    lines = [" | ".join(columns)]
    for row in rows[:limit]:
        lines.append(" | ".join(_cell(value) for value in row))
    if len(rows) > limit:
        lines.append(f"... {len(rows) - limit} more rows")
    return "\n".join(lines)


def fallback_answer(result) -> str:
    if not result.rows:
        return "No rows matched."
    if len(result.columns) == 1 and len(result.rows) == 1:
        return f"{result.columns[0]}: {_cell(result.rows[0][0])}"
    first = ", ".join(
        f"{c} = {_cell(v)}" for c, v in zip(result.columns, result.rows[0], strict=True)
    )
    return f"{len(result.rows)} row(s) returned. First row: {first}."


def _cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)
