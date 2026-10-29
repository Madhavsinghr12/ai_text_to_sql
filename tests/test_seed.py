"""The eval set stores expected answers as SQL, so the fixture itself has to be
reproducible. These tests pin that."""

from __future__ import annotations

import hashlib
import sqlite3

from text2sql.seed import build_database

EXPECTED_COUNTS = {
    "categories": 10,
    "products": 120,
    "customers": 500,
    "orders": 5200,
    "order_items": 12953,
    "returns": 710,
}


def test_row_counts_are_stable(demo_db) -> None:
    with sqlite3.connect(demo_db) as conn:
        actual = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in EXPECTED_COUNTS
        }
    assert actual == EXPECTED_COUNTS


def test_rebuild_is_byte_identical(tmp_path) -> None:
    first = build_database(tmp_path / "a.db")
    second = build_database(tmp_path / "b.db")
    assert (
        hashlib.sha256(first.read_bytes()).hexdigest()
        == hashlib.sha256(second.read_bytes()).hexdigest()
    )


def test_no_orphaned_order_items(demo_db) -> None:
    with sqlite3.connect(demo_db) as conn:
        orphans = conn.execute(
            "SELECT COUNT(*) FROM order_items oi LEFT JOIN orders o ON o.id = oi.order_id "
            "WHERE o.id IS NULL"
        ).fetchone()[0]
    assert orphans == 0
