from contextlib import contextmanager

from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import get_settings

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            get_settings().database_url,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
            configure=register_vector,
            check=ConnectionPool.check_connection,
            open=True,
        )
    return _pool


@contextmanager
def connection():
    with get_pool().connection() as conn:
        yield conn
