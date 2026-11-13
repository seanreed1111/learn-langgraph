"""Who may read and write what. This is data, not code paths: `db.py` turns a `Policy` into a SQLite
*authorizer*, so the database itself enforces it, column by column.

The lesson-20 agent had one global allow-list of tables. A production agent serves several people, and
"the analyst" must not be able to read customer emails just because the support team can.

Real databases do this with GRANTs (Postgres: `GRANT SELECT (id, name) ON customers TO analyst`) and
row-level security. SQLite has neither, so the authorizer plays that role in this demo. The shape of
`Policy` is what carries over.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class Policy:
    readable: Mapping[str, frozenset[str]] = field(default_factory=dict)  # table -> readable columns
    writable: frozenset[str] = frozenset()  # tables that may be modified (after human approval)
    can_approve: bool = False  # may approve another person's pending write
    can_audit: bool = False  # may read the audit log

    def can_read_table(self, table: str) -> bool:
        return table.lower() in self.readable

    def can_read_column(self, table: str, column: str) -> bool:
        cols = self.readable.get(table.lower())
        # "" is how SQLite reports a table read with no named column (count(*), rowid).
        return cols is not None and (column == "" or column.lower() in cols)

    def can_write_table(self, table: str) -> bool:
        return table.lower() in self.writable and self.can_read_table(table)


DENY_ALL = Policy()

_CUSTOMER_PUBLIC = frozenset({"id", "name", "country", "signup_year"})
_PRODUCTS = frozenset({"id", "name", "category", "price"})
_ORDERS = frozenset({"id", "customer_id", "order_year", "status"})
_ORDER_ITEMS = frozenset({"id", "order_id", "product_id", "quantity"})


def _tables(*, customer_email: bool) -> dict[str, frozenset[str]]:
    customers = _CUSTOMER_PUBLIC | {"email"} if customer_email else _CUSTOMER_PUBLIC
    return {"customers": customers, "products": _PRODUCTS, "orders": _ORDERS, "order_items": _ORDER_ITEMS}


ROLES: Mapping[str, Policy] = MappingProxyType({
    # sees the catalogue only
    "viewer": Policy(readable={"products": _PRODUCTS}),
    # business analytics: everything except personal data (customer email)
    "analyst": Policy(readable=_tables(customer_email=False)),
    # customer support: sees emails, may PROPOSE changes to orders (a manager approves them)
    "support": Policy(readable=_tables(customer_email=True), writable=frozenset({"orders"})),
    # may propose changes to orders and customers, approves other people's writes, reads the audit log
    "manager": Policy(readable=_tables(customer_email=True), writable=frozenset({"orders", "customers"}),
                      can_approve=True, can_audit=True),
})


def policy_for(role: str) -> Policy:
    """Unknown role -> DENY_ALL. Fail closed."""
    return ROLES.get(role, DENY_ALL)


@dataclass(frozen=True)
class SqlContext:
    """Per-request, READ-ONLY identity. Set by your auth layer, never by the model or the request body
    (the same rule as `RagContext` in lesson 14). Kept in `context`, so it is not written to checkpoints."""

    user_id: str
    role: str = "viewer"

    @property
    def policy(self) -> Policy:
        return policy_for(self.role)


ANONYMOUS = SqlContext("anonymous", "none")  # what a run with no context gets: it can read nothing
