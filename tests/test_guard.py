"""The guard is the only thing standing between model output and the database,
so it gets the most adversarial tests."""

from __future__ import annotations

import pytest

from text2sql.guard import SQLRejected, validate

TABLES = {"categories", "products", "customers", "orders", "order_items", "returns"}


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM orders",
        "SELECT c.country, SUM(oi.quantity * oi.unit_price) AS revenue "
        "FROM orders o JOIN customers c ON c.id = o.customer_id "
        "JOIN order_items oi ON oi.order_id = o.id "
        "WHERE o.status = 'completed' GROUP BY c.country ORDER BY revenue DESC",
        # CTE: the CTE name must not be mistaken for an unknown table
        "WITH totals AS (SELECT product_id, SUM(quantity) AS q "
        "FROM order_items GROUP BY product_id) "
        "SELECT p.name, totals.q FROM totals JOIN products p ON p.id = totals.product_id",
        "SELECT id FROM orders UNION SELECT id FROM products",
        # a window function over an allowed table
        "SELECT name, RANK() OVER (PARTITION BY category_id ORDER BY unit_price DESC) AS r "
        "FROM products",
    ],
)
def test_allows_read_queries(sql: str) -> None:
    assert validate(sql, TABLES)


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("DROP TABLE orders", "SELECT"),
        ("DELETE FROM orders", "SELECT"),
        ("UPDATE orders SET status = 'completed'", "SELECT"),
        ("INSERT INTO orders (id) VALUES (1)", "SELECT"),
        ("SELECT 1; DROP TABLE orders", "one statement"),
        ("PRAGMA table_info(orders)", "SELECT"),
        ("ATTACH DATABASE 'stolen.db' AS stolen", "SELECT"),
        ("SELECT * FROM secret_table", "Unknown table"),
        ("SELECT * FROM sqlite_master", "Unknown table"),
        ("SELECT load_extension('/tmp/evil')", "load_extension"),
        ("SELECT readfile('/etc/passwd')", "readfile"),
        ("", "no SQL"),
        ("SELEC * FROM orders", "parse"),
    ],
)
def test_rejects_unsafe_or_broken_sql(sql: str, reason: str) -> None:
    with pytest.raises(SQLRejected) as excinfo:
        validate(sql, TABLES)
    # The message is relayed to the model, so it has to say what went wrong.
    assert reason.lower() in str(excinfo.value).lower()


def test_applies_row_cap_when_missing() -> None:
    guarded = validate("SELECT * FROM products", TABLES, max_rows=50)
    assert "LIMIT 50" in guarded


def test_lowers_an_existing_large_limit() -> None:
    guarded = validate("SELECT * FROM products LIMIT 100000", TABLES, max_rows=50)
    assert "LIMIT 50" in guarded
    assert "100000" not in guarded


def test_keeps_a_smaller_limit() -> None:
    guarded = validate("SELECT * FROM products LIMIT 5", TABLES, max_rows=50)
    assert "LIMIT 5" in guarded


def test_preserves_order_by_when_capping() -> None:
    guarded = validate("SELECT name FROM products ORDER BY unit_price DESC", TABLES, max_rows=20)
    assert "ORDER BY" in guarded.upper()
    assert "LIMIT 20" in guarded


def test_unknown_table_message_lists_available_tables() -> None:
    with pytest.raises(SQLRejected) as excinfo:
        validate("SELECT * FROM sales", TABLES)
    message = str(excinfo.value)
    assert "sales" in message
    assert "order_items" in message
