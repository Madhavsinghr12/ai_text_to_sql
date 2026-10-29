"""Accuracy harness.

For every question: run the reference SQL to get the expected result, run the
agent, compare. Comparison is by row position, so column order matters; floats are
rounded before comparison.

    python eval/run_eval.py                  # needs an API key
    python eval/run_eval.py --fake           # no key: proves the harness itself
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text2sql.agent import TextToSQLAgent  # noqa: E402
from text2sql.executor import execute  # noqa: E402
from text2sql.guard import validate  # noqa: E402
from text2sql.llm import OpenAIModel  # noqa: E402
from text2sql.schema import table_names  # noqa: E402
from text2sql.seed import build_database  # noqa: E402
from text2sql.testing import ScriptedModel  # noqa: E402

QUESTIONS_FILE = ROOT / "eval" / "questions.yaml"
DB_FILE = ROOT / "data" / "retail.db"
REPORT_FILE = ROOT / "eval" / "report.json"


def load_questions(path: Path, only: str | None) -> list[dict]:
    questions = yaml.safe_load(path.read_text(encoding="utf-8"))
    if only:
        questions = [q for q in questions if only in q["id"]]
    return questions


def reference_rows(question: dict, db_path: Path, max_rows: int) -> list[tuple]:
    """Expected rows, produced through the same guard and executor as the agent."""
    sql = validate(question["reference_sql"], set(table_names(db_path)), max_rows)
    return execute(db_path, sql, max_rows).rows


def normalise(rows: list[tuple], decimals: int) -> list[tuple]:
    normalised = []
    for row in rows:
        cells = []
        for value in row:
            if isinstance(value, float):
                value = round(value, decimals)
                if value == int(value):
                    value = int(value)
            cells.append(value)
        normalised.append(tuple(cells))
    return normalised


def matches(expected: list[tuple], actual: list[tuple], order_matters: bool, decimals: int) -> bool:
    left = normalise(expected, decimals)
    right = normalise(actual, decimals)
    if order_matters:
        return left == right
    return Counter(left) == Counter(right)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    db_path = Path(args.db)
    if not db_path.exists():
        build_database(db_path)

    questions = load_questions(Path(args.questions), args.only)
    if not questions:
        print("no questions matched", file=sys.stderr)
        return 2

    if args.fake:
        model_label = "fake-scripted"
        make_model = lambda q: ScriptedModel([f"```sql\n{q['reference_sql'].strip()}\n```"])  # noqa: E731
    else:
        try:
            base = OpenAIModel(model=args.model, base_url=args.base_url)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        model_label = base.model
        make_model = lambda q: base  # noqa: E731

    results = []
    for question in questions:
        expected = reference_rows(question, db_path, args.limit)
        agent = TextToSQLAgent(
            make_model(question),
            db_path,
            max_rows=args.limit,
            summarise=False,
            today=date.today(),
        )
        result = agent.ask(question["question"])
        passed = result.ok and matches(
            expected, result.rows, question.get("order_matters", False), args.decimals
        )
        results.append(
            {
                "id": question["id"],
                "difficulty": question.get("difficulty", "unrated"),
                "question": question["question"],
                "passed": passed,
                "attempts": len(result.attempts),
                "elapsed_ms": round(result.elapsed_ms, 1),
                "sql": result.sql,
                "error": result.error or result.clarification,
                "expected_rows": [list(row) for row in expected],
                "actual_rows": [list(row) for row in result.rows],
            }
        )
        _print_row(results[-1], args.verbose)

    passed_count = sum(1 for r in results if r["passed"])
    accuracy = 100.0 * passed_count / len(results)

    print(f"\n{passed_count}/{len(results)} passed ({accuracy:.1f}%)  |  model: {model_label}")
    for difficulty in ("basic", "intermediate", "advanced"):
        subset = [r for r in results if r["difficulty"] == difficulty]
        if subset:
            hits = sum(1 for r in subset if r["passed"])
            print(f"  {difficulty:<13} {hits}/{len(subset)}")

    report = {
        "model": model_label,
        "generated_at": date.today().isoformat(),
        "accuracy": round(accuracy, 1),
        "passed": passed_count,
        "total": len(results),
        "results": results,
    }
    Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {args.json_out}")
    return 0 if passed_count == len(results) else 1


def _print_row(result: dict, verbose: bool) -> None:
    flag = "ok  " if result["passed"] else "FAIL"
    print(
        f"{flag} {result['id']}  {result['elapsed_ms']:>7.0f} ms  "
        f"{result['attempts']}x  {result['question'][:58]}"
    )
    if result["passed"] or not verbose:
        return
    if result["error"]:
        print(f"       error: {result['error']}")
    print(f"       expected: {result['expected_rows'][:3]}")
    print(f"       actual:   {result['actual_rows'][:3]}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure text-to-SQL accuracy on the demo database."
    )
    parser.add_argument("--questions", default=str(QUESTIONS_FILE))
    parser.add_argument("--db", default=str(DB_FILE))
    parser.add_argument("--limit", type=int, default=200, help="row cap used for both sides")
    parser.add_argument("--decimals", type=int, default=2, help="rounding for float comparison")
    parser.add_argument("--only", help="run questions whose id contains this string")
    parser.add_argument("--model", help="model name (default: env T2SQL_MODEL)")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint")
    parser.add_argument("--fake", action="store_true", help="use reference SQL as the model")
    parser.add_argument("--json-out", default=str(REPORT_FILE))
    parser.add_argument("--verbose", action="store_true", help="show expected/actual on failure")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
