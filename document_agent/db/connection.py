from __future__ import annotations

from pathlib import Path
from typing import Optional

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from document_agent.config import Settings, get_settings

_POOL: Optional[ConnectionPool] = None


def get_pool(settings: Optional[Settings] = None) -> ConnectionPool:
    global _POOL
    settings = settings or get_settings()
    if _POOL is None:
        _POOL = ConnectionPool(
            conninfo=settings.database_url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            kwargs={"row_factory": dict_row},
            open=True,
        )
    return _POOL


def close_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None


def init_db(settings: Optional[Settings] = None) -> None:
    settings = settings or get_settings()
    migration = _migration_path()
    sql = migration.read_text(encoding="utf-8")
    pool = get_pool(settings)
    with pool.connection() as conn:
        conn.execute(sql)
        conn.commit()


def _migration_path() -> Path:
    candidates = [
        Path.cwd() / "migrations" / "001_init.sql",
        Path(__file__).resolve().parents[2] / "migrations" / "001_init.sql",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate migrations/001_init.sql")
