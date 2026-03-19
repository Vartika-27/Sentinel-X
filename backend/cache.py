"""
cache.py – Redis-backed result cache for Sentinel-X.

Results are keyed by SHA-256(image_bytes) + epsilon so that identical
requests return instantly without re-running the ML pipeline.
"""

import hashlib
import json
import os
from typing import Any, Optional

import redis.asyncio as aioredis

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

_redis: Optional[aioredis.Redis] = None


async def init_redis() -> None:
    """Open the async Redis connection (called once from FastAPI lifespan)."""
    global _redis
    url = os.getenv("REDIS_URL", "redis://localhost:6379")
    _redis = aioredis.from_url(url, decode_responses=True)
    await _redis.ping()          # fail fast if Redis is unreachable
    print(f"Redis connected: {url}")


async def close_redis() -> None:
    """Close the Redis connection on application shutdown."""
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def compute_image_hash(image_bytes: bytes) -> str:
    """Return hex SHA-256 of raw image bytes (used as cache key component)."""
    return hashlib.sha256(image_bytes).hexdigest()


def _make_key(image_hash: str, epsilon: float) -> str:
    # Normalise epsilon to 4 decimal places to avoid float string drift
    return f"sentinel:{image_hash}:{epsilon:.4f}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_cached(image_hash: str, epsilon: float) -> Optional[dict]:
    """
    Look up a cached attack result.

    Returns the deserialised dict on a cache hit, or None on a miss.
    """
    if _redis is None:
        return None
    key = _make_key(image_hash, epsilon)
    raw = await _redis.get(key)
    if raw is None:
        return None
    return json.loads(raw)


async def set_cached(
    image_hash: str,
    epsilon: float,
    result: Any,
    ttl: int = 3600,
) -> None:
    """
    Store an attack result in Redis.

    Args:
        image_hash: SHA-256 hex string of the image bytes.
        epsilon:    Attack strength used.
        result:     JSON-serialisable result dict.
        ttl:        Time-to-live in seconds (default: 1 hour).
    """
    if _redis is None:
        return
    key = _make_key(image_hash, epsilon)
    await _redis.setex(key, ttl, json.dumps(result))
