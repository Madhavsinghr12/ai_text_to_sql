from __future__ import annotations

from text2sql.schema import introspect, render_schema, table_names

EXPECTED_TABLES = ["categories", "customers", "order_items", "orders", "products", "returns"]


def test_finds_every_user_table(demo_db) -> None:
    assert table_names(demo_db) == EXPECTED_TABLES


def test_columns_carry_primary_keys(demo_db) -> None:
    columns = introspect(demo_db)
    id_column = next(c for c in columns["orders"] if c["name"] == "id")
    assert id_column["pk"] is True


def test_rendered_schema_describes_keys_counts_and_samples(demo_db) -> None:
    text = render_schema(demo_db)
    assert "TABLE orders (5,200 rows)" in text
    assert "customer_id INTEGER  [not null, fk -> customers.id]" in text
    assert "e.g. (" in text
    # the model should not be shown thousands of rows or the raw DDL
    assert "CREATE TABLE" not in text
