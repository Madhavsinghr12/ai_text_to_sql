"""Schema grounding: turn a live database into the compact description the model reads.

The model never sees the whole DDL dump. It sees table names, columns, types,
foreign keys, row counts and a couple of sample rows -- enough to write a correct
query without burning the context window on 6,000 rows of data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SAMPLE_ROWS = 2


def read_only_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open the database through a read-only URI.

    This is the second safety layer: even if a write slipped past the SQL guard,
    SQLite itself refuses to modify a ``mode=ro`` database.
    """
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def table_names(db_path: str | Path) -> list[str]:
    conn = read_only_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows]


def introspect(db_path: str | Path) -> dict[str, list[dict]]:
    """Return ``{table: [column info, ...]}`` for every user table."""
    conn = read_only_connection(db_path)
    try:
        return {
            table: [
                {"name": name, "type": type_, "notnull": bool(notnull), "pk": bool(pk)}
                for _, name, type_, notnull, _default, pk in conn.execute(
                    f"PRAGMA table_info({_quote(table)})"
                )
            ]
            for table in table_names(db_path)
        }
    finally:
        conn.close()


def render_schema(db_path: str | Path, sample_rows: int = SAMPLE_ROWS) -> str:
    """Render the prompt-facing schema description."""
    conn = read_only_connection(db_path)
    try:
        foreign_keys = _foreign_keys(conn, table_names(db_path))
        blocks = []
        for table in table_names(db_path):
            count = conn.execute(f"SELECT COUNT(*) FROM {_quote(table)}").fetchone()[0]
            columns = conn.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
            lines = [f"TABLE {table} ({count:,} rows)"]
            for _cid, name, type_, notnull, _default, pk in columns:
                notes = []
                if pk:
                    notes.append("pk")
                if notnull:
                    notes.append("not null")
                if (table, name) in foreign_keys:
                    notes.append(f"fk -> {foreign_keys[(table, name)]}")
                suffix = f"  [{', '.join(notes)}]" if notes else ""
                lines.append(f"  {name} {type_}{suffix}")
            for row in conn.execute(f"SELECT * FROM {_quote(table)} LIMIT {int(sample_rows)}"):
                lines.append(f"  e.g. {_format_row(row)}")
            blocks.append("\n".join(lines))
    finally:
        conn.close()
    return "\n\n".join(blocks)


def _foreign_keys(conn: sqlite3.Connection, tables: list[str]) -> dict[tuple[str, str], str]:
    keys: dict[tuple[str, str], str] = {}
    for table in tables:
        for row in conn.execute(f"PRAGMA foreign_key_list({_quote(table)})"):
            _id, _seq, target_table, from_col, to_col, *_ = row
            keys[(table, from_col)] = f"{target_table}.{to_col}"
    return keys


def _format_row(row: tuple) -> str:
    return "(" + ", ".join(repr(value) for value in row) + ")"


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'
