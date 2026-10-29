"""Command line interface.

python -m text2sql "which 5 products made the most revenue in September?"
python -m text2sql                      # interactive
python -m text2sql --show-schema        # what the model sees
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from tabulate import tabulate

from text2sql.agent import Result, TextToSQLAgent
from text2sql.llm import OpenAIModel
from text2sql.schema import render_schema
from text2sql.seed import build_database

DEFAULT_DB = Path("data/retail.db")
MAX_CELL = 42


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    db_path = Path(args.db)

    if args.reseed or not db_path.exists():
        print(f"seeding demo database at {db_path} ...", file=sys.stderr)
        build_database(db_path, force=args.reseed)

    if args.show_schema:
        print(render_schema(db_path))
        if not args.question:
            return 0

    try:
        model = OpenAIModel(model=args.model, base_url=args.base_url)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    agent = TextToSQLAgent(
        model,
        db_path,
        max_rows=args.limit,
        summarise=not args.no_summary,
    )

    if args.question:
        return _report(agent.ask(args.question), args)

    print("Ask a question about the store. Empty line or 'exit' to quit.")
    while True:
        try:
            question = input("\nquestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question or question.lower() in {"exit", "quit"}:
            return 0
        _report(agent.ask(question), args)


def _report(result: Result, args: argparse.Namespace) -> int:
    if args.json:
        payload = asdict(result)
        payload["ok"] = result.ok
        print(json.dumps(payload, indent=2, default=str))
        return 0 if result.ok else 1

    if result.clarification:
        print(f"\nClarification needed: {result.clarification}")
        return 1

    if result.error:
        print(f"\nerror: {result.error}", file=sys.stderr)
        for attempt in result.attempts:
            print(f"  - {attempt.error}", file=sys.stderr)
        return 1

    print(f"\nSQL\n{_indent(result.sql or '')}")
    if result.rows:
        table = [[_clip(value) for value in row] for row in result.rows]
        rendered = tabulate(
            table, headers=result.columns, tablefmt="simple", floatfmt=",.2f", intfmt=","
        )
        print(f"\n{rendered}")
    else:
        print("\n(no rows)")

    print(f"\n{result.answer}")
    footer = (
        f"{len(result.rows)} row(s) | {result.elapsed_ms:.0f} ms | "
        f"{len(result.attempts)} attempt(s)"
    )
    if result.truncated:
        footer += " | result truncated"
    print(f"\n{footer}")
    return 0


def _indent(text: str) -> str:
    return "\n".join("  " + line for line in text.splitlines())


def _clip(value: object) -> object:
    text = str(value)
    return text if len(text) <= MAX_CELL else text[: MAX_CELL - 3] + "..."


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="text2sql",
        description="Ask a business question in plain English; get validated read-only SQL back.",
    )
    parser.add_argument("question", nargs="?", help="one question; omit for interactive mode")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database path")
    parser.add_argument("--limit", type=int, default=200, help="maximum rows returned")
    parser.add_argument("--model", help="model name (default: env T2SQL_MODEL)")
    parser.add_argument(
        "--base-url", help="OpenAI-compatible endpoint (default: env T2SQL_BASE_URL)"
    )
    parser.add_argument(
        "--show-schema", action="store_true", help="print the schema the model sees"
    )
    parser.add_argument("--no-summary", action="store_true", help="skip the second model call")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--reseed", action="store_true", help="rebuild the demo database first")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
