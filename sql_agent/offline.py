"""Offline SQL writer/answerer (a TEST DOUBLE).

It maps a handful of question shapes to SQL with rules. It exists so the *rails* (guard,
dry-run, approval, repair loop, evals) are runnable and deterministic without an API key.
It is deliberately imperfect: `revenue` questions start with a wrong query so the
repair loop has something to repair. It also happily emits dangerous SQL for hostile
prompts, to prove the guard - not the model - is what keeps you safe.
"""

from __future__ import annotations

import re
from typing import Any

NO_QUERY = ""  # the writer returns this when the question is not answerable from the schema


class OfflineWriter:
    def draft(self, question: str, schema: dict[str, str], error: str | None = None,
              prev_sql: str | None = None) -> str:
        q = question.lower().strip().rstrip("?.! ")

        # ---- hostile prompts: a naive/obedient model would comply
        if re.search(r"\bdrop\b.*\btable\b", q):
            m = re.search(r"table\s+(\w+)|(\w+)\s+table", q)
            return f"DROP TABLE {next(g for g in m.groups() if g) if m else 'customers'}"
        if "ignore" in q and ("sqlite_master" in q or "instructions" in q):
            return "SELECT * FROM sqlite_master"
        if "pragma" in q:
            return "PRAGMA table_info(customers)"
        if "then delete all orders" in q:
            return "SELECT * FROM customers; DELETE FROM orders"

        # ---- writes
        if m := re.search(r"cancel order (\d+)", q):
            return f"UPDATE orders SET status = 'cancelled' WHERE id = {m.group(1)}"
        if m := re.search(r"delete customer (\d+)", q):
            return f"DELETE FROM customers WHERE id = {m.group(1)}"
        if re.search(r"delete all orders|clear (the )?orders", q):
            return "DELETE FROM orders"

        # ---- reads
        if "how many customers" in q:
            return "SELECT COUNT(*) AS customers FROM customers"
        if m := re.search(r"customers (?:are )?from (?:the )?(\w+)", q):
            return f"SELECT name FROM customers WHERE country = '{_country(m.group(1))}' ORDER BY name"
        if m := re.search(r"how many orders were (\w+) in (\d{4})", q):
            return (f"SELECT COUNT(*) AS orders FROM orders "
                    f"WHERE status = '{m.group(1)}' AND order_year = {m.group(2)}")
        if "most expensive product" in q:
            return "SELECT name, price FROM products ORDER BY price DESC LIMIT 1"
        if "revenue" in q and "category" in q:
            if error is not None and "no such column" in error:  # REPAIR using the error text
                return ("SELECT p.category, SUM(oi.quantity * p.price) AS revenue "
                        "FROM order_items oi JOIN products p ON p.id = oi.product_id "
                        "GROUP BY p.category ORDER BY revenue DESC")
            # first attempt: forgets that price lives in `products`
            return ("SELECT category, SUM(quantity * price) AS revenue "
                    "FROM order_items GROUP BY category")
        if "orders has each customer" in q or "orders per customer" in q:
            return ("SELECT c.name, COUNT(o.id) AS orders FROM customers c "
                    "LEFT JOIN orders o ON o.customer_id = c.id GROUP BY c.id ORDER BY c.name")
        if m := re.search(r"products in the (\w+) category", q):
            return f"SELECT name FROM products WHERE category = '{m.group(1)}' ORDER BY name"
        if "which country has the most customers" in q:
            return ("SELECT country, COUNT(*) AS n FROM customers "
                    "GROUP BY country ORDER BY n DESC LIMIT 1")
        if "average product price" in q:
            return "SELECT ROUND(AVG(price), 2) AS avg_price FROM products"
        if "customers have no orders" in q:
            return ("SELECT name FROM customers WHERE id NOT IN "
                    "(SELECT customer_id FROM orders) ORDER BY name")
        return NO_QUERY

    def answer(self, question: str, sql: str, cols: list[str], rows: list[tuple[Any, ...]]) -> str:
        if not rows:
            return "No matching rows."
        if len(rows) == 1 and len(rows[0]) == 1:
            return f"{cols[0]}: {rows[0][0]}"
        return "\n".join(", ".join(f"{c}={v}" for c, v in zip(cols, r, strict=True)) for r in rows)


def _country(word: str) -> str:
    return {"uk": "UK", "usa": "USA", "us": "USA", "finland": "Finland"}.get(word, word.title())
