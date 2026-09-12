"""Database access. Thin psycopg3 wrapper with a connection pool.

No ORM on purpose. The interesting queries here are hybrid-search queries that
mix a vector distance operator, a tsvector rank, and a relational filter in one
statement -- expressing those through an ORM costs clarity and buys nothing.
"""

from __future__ import annotations

import atexit
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import get_settings

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        s = get_settings()
        _pool = ConnectionPool(
            s.db_dsn,
            min_size=1,
            max_size=8,
            kwargs={"row_factory": dict_row},
            open=True,
        )
        # Python 3.14 raises PythonFinalizationError if the pool's worker
        # threads are still joinable when the interpreter tears down, which
        # prints an alarming traceback after an otherwise successful command.
        # Closing at exit is tidier than letting __del__ race finalization.
        atexit.register(close_pool)
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    with get_pool().connection() as conn:
        register_vector(conn)
        yield conn


def register_vector(conn: psycopg.Connection) -> None:
    """Teach psycopg how to adapt pgvector values, once per connection."""
    if getattr(conn, "_edgar_vector_registered", False):
        return
    try:
        from pgvector.psycopg import register_vector as _rv

        _rv(conn)
    except Exception:
        # pgvector's adapter needs the extension present. Tests that never
        # touch embeddings can proceed without it.
        pass
    conn._edgar_vector_registered = True  # type: ignore[attr-defined]


def query(sql: str, params: Sequence[Any] | dict[str, Any] | None = None) -> list[dict[str, Any]]:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        if cur.description is None:
            return []
        return list(cur.fetchall())


def query_one(
    sql: str, params: Sequence[Any] | dict[str, Any] | None = None
) -> dict[str, Any] | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: Sequence[Any] | dict[str, Any] | None = None) -> int:
    with connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


def execute_many(sql: str, rows: Sequence[Sequence[Any]]) -> int:
    if not rows:
        return 0
    with connection() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        return cur.rowcount


def apply_schema(path: str = "sql/001_schema.sql") -> None:
    with open(path, encoding="utf-8") as fh:
        ddl = fh.read()
    with connection() as conn, conn.cursor() as cur:
        cur.execute(ddl)


def jsonb(value: Any) -> str:
    """psycopg needs dicts serialised before they can go into a jsonb column."""
    return json.dumps(value, default=str)


def healthcheck() -> bool:
    try:
        row = query_one("SELECT 1 AS ok")
        return bool(row and row.get("ok") == 1)
    except Exception:
        return False
