"""Demo database: a tiny e-commerce schema, deterministic, created from code.

Two connection kinds - the core safety idea of the lesson:
  * `connect_readonly` : SQLite opened `mode=ro` PLUS an authorizer that only permits reads.
                         Even if the SQL guard had a bug, the database itself refuses writes.
  * `connect_writer`   : used ONLY by the approved-write node after a human said yes.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE customers (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT NOT NULL, signup_year INTEGER NOT NULL);
CREATE TABLE products (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL, price REAL NOT NULL);
CREATE TABLE orders (
  id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
  order_year INTEGER NOT NULL, status TEXT NOT NULL);
CREATE TABLE order_items (
  id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL REFERENCES orders(id),
  product_id INTEGER NOT NULL REFERENCES products(id), quantity INTEGER NOT NULL);
"""

CUSTOMERS = [
    (1, "Ada Lovelace", "UK", 2021), (2, "Grace Hopper", "USA", 2020),
    (3, "Alan Turing", "UK", 2022), (4, "Katherine Johnson", "USA", 2022),
    (5, "Linus Torvalds", "Finland", 2023), (6, "Margaret Hamilton", "USA", 2023),
]
PRODUCTS = [
    (1, "Mechanical Keyboard", "peripherals", 120.0), (2, "Wireless Mouse", "peripherals", 45.0),
    (3, "4K Monitor", "displays", 420.0), (4, "Laptop Stand", "accessories", 60.0),
    (5, "USB-C Hub", "accessories", 35.0), (6, "Webcam", "peripherals", 80.0),
]
ORDERS = [
    (1, 1, 2023, "shipped"), (2, 2, 2023, "shipped"), (3, 2, 2024, "shipped"),
    (4, 3, 2024, "cancelled"), (5, 4, 2024, "shipped"), (6, 5, 2024, "pending"),
    (7, 1, 2024, "shipped"), (8, 6, 2024, "shipped"),
]
ORDER_ITEMS = [
    (1, 1, 1, 1), (2, 1, 2, 2), (3, 2, 3, 1), (4, 3, 1, 1), (5, 3, 4, 2),
    (6, 4, 3, 1), (7, 5, 6, 3), (8, 6, 5, 4), (9, 7, 3, 2), (10, 8, 2, 1), (11, 8, 5, 1),
]


def create_demo_db(path: str | Path) -> Path:
    path = Path(path)
    con = sqlite3.connect(path)
    # reset IN PLACE (never unlink: on Windows an open read-only connection would block deletion)
    con.executescript("DROP TABLE IF EXISTS order_items; DROP TABLE IF EXISTS orders; "
                      "DROP TABLE IF EXISTS products; DROP TABLE IF EXISTS customers;")
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO customers VALUES (?,?,?,?)", CUSTOMERS)
    con.executemany("INSERT INTO products VALUES (?,?,?,?)", PRODUCTS)
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)", ORDERS)
    con.executemany("INSERT INTO order_items VALUES (?,?,?,?)", ORDER_ITEMS)
    con.commit()
    con.close()
    return path


_ALLOWED = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
            sqlite3.SQLITE_RECURSIVE}


def _read_only_authorizer(action, arg1, arg2, db_name, source):  # noqa: ANN001
    return sqlite3.SQLITE_OK if action in _ALLOWED else sqlite3.SQLITE_DENY


def connect_readonly(path: str | Path) -> sqlite3.Connection:
    """Layer 1: OS-level read-only open. Layer 2: authorizer denies everything but reads
    (no ATTACH, PRAGMA, INSERT, DDL, load_extension...)."""
    uri = f"file:{Path(path).resolve().as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True, check_same_thread=False)
    con.set_authorizer(_read_only_authorizer)
    return con


def connect_writer(path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(path, check_same_thread=False)


def schema_text(path: str | Path) -> dict[str, str]:
    """table -> CREATE statement. Read directly (trusted code path), not via the agent."""
    con = sqlite3.connect(path)
    try:
        return {n: s for n, s in con.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    finally:
        con.close()


def run_read(con: sqlite3.Connection, sql: str) -> tuple[list[str], list[tuple]]:
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description or []]
    return cols, cur.fetchall()


def dry_run(con: sqlite3.Connection, sql: str) -> None:
    """EXPLAIN compiles the statement (syntax, tables, columns) without executing it."""
    con.execute(f"EXPLAIN {sql}").fetchall()


def preview_write(path: str | Path, sql: str) -> int:
    """Run a write inside a transaction, count affected rows, ROLL BACK. Nothing persists."""
    con = connect_writer(path)
    try:
        con.execute("BEGIN")
        n = con.execute(sql).rowcount
        con.execute("ROLLBACK")
        return n
    finally:
        con.close()


def apply_write(path: str | Path, sql: str) -> int:
    con = connect_writer(path)
    try:
        n = con.execute(sql).rowcount
        con.commit()
        return n
    finally:
        con.close()
