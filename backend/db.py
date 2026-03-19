"""
db.py – PostgreSQL integration for Sentinel-X.

Persists every attack run to the `attack_logs` table so researchers can
query historical data (e.g., "which epsilon values most often flip the
prediction?"). Uses asyncpg for non-blocking database I/O.
"""

import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import asyncpg

# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------

_pool: Optional[asyncpg.Pool] = None


async def init_db() -> None:
    """
    Create the asyncpg connection pool and ensure the `attack_logs` table
    exists. Called once from FastAPI lifespan.
    """
    global _pool
    dsn = os.getenv("POSTGRES_DSN", "postgresql://postgres:password@localhost:5432/sentinelx")
    _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
    if _pool is None:
        raise RuntimeError("asyncpg.create_pool returned None — check POSTGRES_DSN.")
    async with _pool.acquire() as conn:
        await conn.execute(_CREATE_TABLE_SQL)
    print("PostgreSQL connected and schema ready.")


async def close_db() -> None:
    """Close the connection pool on application shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS attack_logs (
    id              SERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ        DEFAULT NOW(),
    filename        TEXT,
    epsilon         DOUBLE PRECISION,
    image_hash      TEXT,
    orig_class      INTEGER,
    orig_label      TEXT,
    orig_conf       DOUBLE PRECISION,
    fgsm_class      INTEGER,
    fgsm_label      TEXT,
    fgsm_conf       DOUBLE PRECISION,
    pgd_class       INTEGER,
    pgd_label       TEXT,
    pgd_conf        DOUBLE PRECISION,
    fgsm_flipped    BOOLEAN,
    pgd_flipped     BOOLEAN,
    cache_hit       BOOLEAN            DEFAULT FALSE
);
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def log_attack(
    filename: str,
    epsilon: float,
    image_hash: str,
    orig: Dict[str, Any],
    fgsm: Dict[str, Any],
    pgd: Dict[str, Any],
    cache_hit: bool = False,
) -> None:
    """
    Insert one row into attack_logs.  Fire-and-forget — failures are logged
    to stderr but never bubble up to the HTTP caller.
    """
    if _pool is None:
        return

    fgsm_flipped = fgsm["class_index"] != orig["class_index"]
    pgd_flipped  = pgd["class_index"]  != orig["class_index"]

    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO attack_logs (
                    filename, epsilon, image_hash,
                    orig_class, orig_label, orig_conf,
                    fgsm_class, fgsm_label, fgsm_conf,
                    pgd_class,  pgd_label,  pgd_conf,
                    fgsm_flipped, pgd_flipped, cache_hit
                ) VALUES (
                    $1,  $2,  $3,
                    $4,  $5,  $6,
                    $7,  $8,  $9,
                    $10, $11, $12,
                    $13, $14, $15
                )
                """,
                filename, epsilon, image_hash,
                orig["class_index"], orig.get("label", ""), orig["confidence"],
                fgsm["class_index"], fgsm.get("label", ""), fgsm["confidence"],
                pgd["class_index"],  pgd.get("label", ""),  pgd["confidence"],
                fgsm_flipped, pgd_flipped, cache_hit,
            )
    except Exception as exc:
        # Non-fatal — never fail the HTTP response due to DB issues
        import sys
        print(f"[db] log_attack failed: {exc}", file=sys.stderr)
