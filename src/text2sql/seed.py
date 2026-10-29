"""Deterministic e-commerce dataset behind the demo and the evaluation set.

Every run produces identical data: the seed is fixed and the reference date is a
constant rather than ``date.today()``. That is what lets ``eval/questions.yaml``
store expected results without the answers drifting over time.
"""

from __future__ import annotations

import math
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

RANDOM_SEED = 42
AS_OF = date(2026, 10, 28)
DAYS_OF_HISTORY = 640
N_CUSTOMERS = 500
N_ORDERS = 5200

CATEGORIES = [
    "Audio",
    "Cameras",
    "Computers",
    "Fitness",
    "Home",
    "Kitchen",
    "Lighting",
    "Office",
    "Outdoors",
    "Toys",
]

PRODUCT_NOUNS = {
    "Audio": ["Headphones", "Earbuds", "Speaker", "Soundbar", "Microphone", "Turntable"],
    "Cameras": ["Camera", "Lens", "Tripod", "Flash", "Gimbal", "Memory Card"],
    "Computers": ["Laptop", "Monitor", "Keyboard", "Mouse", "Docking Station", "Webcam"],
    "Fitness": [
        "Yoga Mat",
        "Kettlebell",
        "Resistance Band",
        "Foam Roller",
        "Jump Rope",
        "Treadmill",
    ],
    "Home": ["Desk Lamp", "Throw Blanket", "Shelving Unit", "Storage Bin", "Wall Clock", "Rug"],
    "Kitchen": ["Blender", "Coffee Grinder", "Kettle", "Cutting Board", "Skillet", "Scale"],
    "Lighting": ["Floor Lamp", "Pendant Light", "Bulb Pack", "Lantern", "LED Strip", "Task Light"],
    "Office": [
        "Desk Chair",
        "Standing Desk",
        "Notebook",
        "Pen Set",
        "Filing Tray",
        "Paper Shredder",
    ],
    "Outdoors": ["Backpack", "Water Bottle", "Headlamp", "Tent", "Hammock", "Cooler"],
    "Toys": ["Building Set", "Board Game", "Puzzle", "Plush Toy", "Model Kit", "Cards Deck"],
}

BRANDS = [
    "Nordic",
    "Vela",
    "Kestrel",
    "Aster",
    "Brightline",
    "Mesa",
    "Onyx",
    "Pike",
    "Quill",
    "Halcyon",
]

# Price band per SKU type. A category-level band would price a mouse like a laptop,
# which shows up immediately in any ranking output a human reads.
NOUN_PRICE_RANGE = {
    # Audio
    "Headphones": (49, 349),
    "Earbuds": (29, 199),
    "Speaker": (39, 299),
    "Soundbar": (79, 499),
    "Microphone": (59, 299),
    "Turntable": (149, 599),
    # Cameras
    "Camera": (199, 1299),
    "Lens": (149, 899),
    "Tripod": (25, 179),
    "Flash": (39, 249),
    "Gimbal": (89, 349),
    "Memory Card": (12, 89),
    # Computers
    "Laptop": (499, 1699),
    "Monitor": (129, 799),
    "Keyboard": (29, 179),
    "Mouse": (15, 89),
    "Docking Station": (79, 249),
    "Webcam": (29, 149),
    # Fitness
    "Yoga Mat": (19, 79),
    "Kettlebell": (29, 149),
    "Resistance Band": (9, 29),
    "Foam Roller": (15, 49),
    "Jump Rope": (8, 25),
    "Treadmill": (599, 1999),
    # Home
    "Desk Lamp": (19, 89),
    "Throw Blanket": (25, 119),
    "Shelving Unit": (59, 249),
    "Storage Bin": (12, 59),
    "Wall Clock": (15, 69),
    "Rug": (49, 299),
    # Kitchen
    "Blender": (39, 249),
    "Coffee Grinder": (29, 149),
    "Kettle": (25, 99),
    "Cutting Board": (15, 59),
    "Skillet": (29, 149),
    "Scale": (15, 49),
    # Lighting
    "Floor Lamp": (49, 199),
    "Pendant Light": (39, 159),
    "Bulb Pack": (9, 29),
    "Lantern": (15, 59),
    "LED Strip": (12, 49),
    "Task Light": (19, 79),
    # Office
    "Desk Chair": (99, 599),
    "Standing Desk": (199, 899),
    "Notebook": (5, 19),
    "Pen Set": (9, 39),
    "Filing Tray": (12, 39),
    "Paper Shredder": (49, 199),
    # Outdoors
    "Backpack": (39, 199),
    "Water Bottle": (12, 39),
    "Headlamp": (15, 49),
    "Tent": (149, 699),
    "Hammock": (39, 149),
    "Cooler": (29, 129),
    # Toys
    "Building Set": (29, 149),
    "Board Game": (19, 69),
    "Puzzle": (12, 39),
    "Plush Toy": (9, 29),
    "Model Kit": (15, 59),
    "Cards Deck": (8, 25),
}

FIRST_NAMES = [
    "Ava",
    "Liam",
    "Noah",
    "Mia",
    "Ethan",
    "Sofia",
    "Lucas",
    "Emma",
    "Oliver",
    "Isla",
    "Mateo",
    "Zara",
    "Hiro",
    "Amara",
    "Nina",
    "Omar",
    "Priya",
    "Diego",
    "Elena",
    "Jonas",
    "Mei",
    "Rafael",
    "Sara",
    "Tariq",
    "Yara",
]

LAST_NAMES = [
    "Chen",
    "Patel",
    "Garcia",
    "Muller",
    "Okafor",
    "Novak",
    "Silva",
    "Haddad",
    "Kowalski",
    "Nguyen",
    "Rossi",
    "Sato",
    "Ibrahim",
    "Larsen",
    "Costa",
    "Weber",
    "Dubois",
    "Kim",
    "Andersen",
    "Reyes",
]

COUNTRIES = [
    ("United States", 0.40),
    ("United Kingdom", 0.14),
    ("Germany", 0.12),
    ("Canada", 0.10),
    ("Australia", 0.08),
    ("India", 0.07),
    ("Brazil", 0.05),
    ("Japan", 0.04),
]

RETURN_REASONS = ["damaged", "wrong item", "not as described", "changed mind", "late delivery"]

SCHEMA_SQL = """
CREATE TABLE categories (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE products (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    category_id INTEGER NOT NULL REFERENCES categories(id),
    unit_price  REAL NOT NULL,
    cost        REAL NOT NULL,
    active      INTEGER NOT NULL
);

CREATE TABLE customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    country     TEXT NOT NULL,
    signup_date TEXT NOT NULL
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date  TEXT NOT NULL,
    status      TEXT NOT NULL,
    channel     TEXT NOT NULL
);

CREATE TABLE order_items (
    id         INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL,
    unit_price REAL NOT NULL
);

CREATE TABLE returns (
    id            INTEGER PRIMARY KEY,
    order_item_id INTEGER NOT NULL REFERENCES order_items(id),
    return_date   TEXT NOT NULL,
    reason        TEXT NOT NULL
);

CREATE INDEX idx_orders_date    ON orders(order_date);
CREATE INDEX idx_orders_status  ON orders(status);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_items_order    ON order_items(order_id);
CREATE INDEX idx_items_product  ON order_items(product_id);
CREATE INDEX idx_returns_item   ON returns(order_item_id);
"""


def build_database(db_path: str | Path, force: bool = False) -> Path:
    """Create the demo database if it is missing. Returns the path."""
    db_path = Path(db_path)
    if db_path.exists() and not force:
        return db_path

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    rng = random.Random(RANDOM_SEED)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        category_ids = _load_categories(conn)
        products = _load_products(conn, rng, category_ids)
        customer_ids = _load_customers(conn, rng)
        _load_orders(conn, rng, customer_ids, products)
        conn.commit()
    finally:
        conn.close()
    return db_path


def _load_categories(conn: sqlite3.Connection) -> list[int]:
    conn.executemany("INSERT INTO categories (name) VALUES (?)", [(c,) for c in CATEGORIES])
    return [row[0] for row in conn.execute("SELECT id FROM categories ORDER BY id")]


def _load_products(
    conn: sqlite3.Connection, rng: random.Random, category_ids: list[int]
) -> list[tuple[int, float]]:
    rows = []
    for category, category_id in zip(CATEGORIES, category_ids, strict=True):
        for noun in PRODUCT_NOUNS[category]:
            low, high = NOUN_PRICE_RANGE[noun]
            for brand in rng.sample(BRANDS, 2):
                price = math.floor(rng.uniform(low, high)) + 0.99
                cost = round(price * rng.uniform(0.45, 0.75), 2)
                active = 0 if rng.random() < 0.08 else 1
                rows.append((f"{brand} {noun}", category_id, price, cost, active))

    conn.executemany(
        "INSERT INTO products (name, category_id, unit_price, cost, active) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    return [
        (product_id, price)
        for product_id, price in conn.execute("SELECT id, unit_price FROM products ORDER BY id")
    ]


def _load_customers(conn: sqlite3.Connection, rng: random.Random) -> list[int]:
    rows = []
    for i in range(N_CUSTOMERS):
        first = rng.choice(FIRST_NAMES)
        last = rng.choice(LAST_NAMES)
        country = rng.choices([c for c, _ in COUNTRIES], weights=[w for _, w in COUNTRIES], k=1)[0]
        signup = AS_OF - timedelta(days=rng.randint(30, 900))
        email = f"{first.lower()}.{last.lower()}{i}@example.com"
        rows.append((f"{first} {last}", email, country, signup.isoformat()))

    conn.executemany(
        "INSERT INTO customers (name, email, country, signup_date) VALUES (?, ?, ?, ?)", rows
    )
    return [row[0] for row in conn.execute("SELECT id FROM customers ORDER BY id")]


def _load_orders(
    conn: sqlite3.Connection,
    rng: random.Random,
    customer_ids: list[int],
    products: list[tuple[int, float]],
) -> None:
    statuses, status_weights = (
        ["completed", "cancelled", "refunded", "pending"],
        [0.82, 0.07, 0.06, 0.05],
    )
    channels, channel_weights = ["web", "mobile", "store"], [0.55, 0.30, 0.15]
    quantities, quantity_weights = [1, 2, 3], [0.70, 0.22, 0.08]
    discounts, discount_weights = [0.0, 0.05, 0.10, 0.15], [0.75, 0.12, 0.09, 0.04]

    # Zipf-ish popularity: rank the catalogue on a shuffled draw, then decay
    # smoothly. A bestseller outsells the long tail about 8:1 -- realistic for a
    # 120-SKU catalogue, and it keeps ranking questions non-degenerate.
    ranked = list(range(len(products)))
    rng.shuffle(ranked)
    popularity = [0.0] * len(products)
    for rank, index in enumerate(ranked):
        popularity[index] = 1.0 / (1.0 + 0.055 * rank)

    item_rows = []
    for order_id in range(1, N_ORDERS + 1):
        order_date = AS_OF - timedelta(days=rng.randint(0, DAYS_OF_HISTORY - 1))
        status = rng.choices(statuses, weights=status_weights, k=1)[0]
        channel = rng.choices(channels, weights=channel_weights, k=1)[0]
        customer_id = rng.choice(customer_ids)
        conn.execute(
            "INSERT INTO orders (id, customer_id, order_date, status, channel) "
            "VALUES (?, ?, ?, ?, ?)",
            (order_id, customer_id, order_date.isoformat(), status, channel),
        )

        chosen = rng.choices(products, weights=popularity, k=rng.randint(1, 4))
        seen: set[int] = set()
        for product_id, price in chosen:
            if product_id in seen:
                continue
            seen.add(product_id)
            discount = rng.choices(discounts, weights=discount_weights, k=1)[0]
            item_rows.append(
                (
                    order_id,
                    product_id,
                    rng.choices(quantities, weights=quantity_weights, k=1)[0],
                    round(price * (1 - discount), 2),
                    status,
                    order_date,
                )
            )

    return_rows = []
    for row in item_rows:
        order_id, product_id, quantity, unit_price, status, order_date = row
        cur = conn.execute(
            "INSERT INTO order_items (order_id, product_id, quantity, unit_price) "
            "VALUES (?, ?, ?, ?)",
            (order_id, product_id, quantity, unit_price),
        )
        if status == "completed" and rng.random() < 0.06:
            return_date = order_date + timedelta(days=rng.randint(3, 30))
            return_rows.append((cur.lastrowid, return_date.isoformat(), rng.choice(RETURN_REASONS)))

    conn.executemany(
        "INSERT INTO returns (order_item_id, return_date, reason) VALUES (?, ?, ?)", return_rows
    )


if __name__ == "__main__":
    path = build_database(Path(__file__).resolve().parents[2] / "data" / "retail.db", force=True)
    with sqlite3.connect(path) as conn:
        for table in ("categories", "products", "customers", "orders", "order_items", "returns"):
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table:<12} {count:>7,} rows")
    print(f"\nwrote {path}")
