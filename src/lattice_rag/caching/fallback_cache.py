from __future__ import annotations

import datetime
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pybreaker
import structlog
from redis.asyncio import Redis

logger = structlog.get_logger(__name__)

REDIS_HASH_KEY = "cache:fallback:queries"


class FallbackCache:
    """Tier 2 Circuit Breaker + Embedded In-Process Redis Hash fallback."""

    def __init__(self, redis_url: str = "embedded") -> None:
        """Sets up Redis async client and pybreaker CircuitBreaker.

        Args:
            redis_url: URL for Redis. Default is 'embedded' which uses in-process
                       fakeredis, requiring zero external server or API keys.
        """
        self.redis_url = redis_url
        self.breaker = pybreaker.CircuitBreaker(fail_max=3, reset_timeout=30)
        self.redis: Redis | None = None
        self._cached_queries: list[str] = []

    async def connect(self) -> None:
        """Establishes Redis connection (in-process fakeredis by default)."""
        if self.redis is not None:
            return

        if self.redis_url == "embedded" or self.redis_url.startswith("fakeredis"):
            import fakeredis.aioredis

            self.redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
            logger.info("embedded_redis_connected", mode="fakeredis")
        else:
            self.redis = Redis.from_url(self.redis_url, decode_responses=True)
            logger.info("external_redis_connected", url=self.redis_url)

    async def _ensure_connected(self) -> None:
        if self.redis is None:
            await self.connect()

    async def call_with_breaker(
        self, coro_fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Executes an async callable under the circuit breaker.

        If the circuit breaker is OPEN, immediately raises pybreaker.CircuitBreakerError
        without executing coro_fn.
        If coro_fn raises an exception, increments the failure counter and trips
        to OPEN when fail_max is reached.
        If coro_fn succeeds, resets failure counter and recovers from half-open.
        """
        timeout = datetime.timedelta(seconds=self.breaker.reset_timeout)
        opened_at = self.breaker._state_storage.opened_at
        if self.breaker.current_state == "open":
            if opened_at and datetime.datetime.now(datetime.timezone.utc) < opened_at + timeout:
                raise pybreaker.CircuitBreakerError("Circuit breaker is OPEN")
            self.breaker.half_open()

        for listener in self.breaker.listeners:
            listener.before_call(self.breaker, coro_fn, *args, **kwargs)

        try:
            result = await coro_fn(*args, **kwargs)
            self.breaker._state_storage.reset_counter()
            self.breaker.state.on_success()
            for listener in self.breaker.listeners:
                listener.success(self.breaker)
            return result
        except BaseException as e:
            if self.breaker.is_system_error(e):
                self.breaker._inc_counter()
                for listener in self.breaker.listeners:
                    listener.failure(self.breaker, e)
                self.breaker.state.on_failure(e)
            raise

    async def get_fallback(self, query_hash: str) -> str | None:
        """Looks up cache:fallback:queries hash in Redis by query_hash or raw query string."""
        await self._ensure_connected()
        assert self.redis is not None

        val = await self.redis.hget(REDIS_HASH_KEY, query_hash)  # type: ignore
        if val is not None:
            logger.debug("fallback_hit_direct", key=query_hash)
            return val

        # Try sha256 hash of normalized input
        sha = hashlib.sha256(query_hash.strip().lower().encode("utf-8")).hexdigest()
        val = await self.redis.hget(REDIS_HASH_KEY, sha)  # type: ignore
        if val is not None:
            logger.debug("fallback_hit_sha", key=sha)
            return val

        # Try raw normalized text
        norm = query_hash.strip().lower()
        val = await self.redis.hget(REDIS_HASH_KEY, norm)  # type: ignore
        if val is not None:
            logger.debug("fallback_hit_norm", key=norm)
            return val

        return None

    async def find_closest_fallback(self, query: str) -> str | None:
        """Finds closest cached query using fuzzy string matching and returns fallback answer."""
        await self._ensure_connected()
        if not self._cached_queries:
            # Populate query list from redis keys if empty
            assert self.redis is not None
            all_keys = await self.redis.hkeys(REDIS_HASH_KEY)  # type: ignore
            # Filter keys that are human-readable (not 64-char sha256 hashes)
            self._cached_queries = [k for k in all_keys if len(k) != 64]

        if not self._cached_queries:
            return None

        norm_query = query.strip().lower()
        matches = difflib.get_close_matches(norm_query, self._cached_queries, n=1, cutoff=0.5)
        if matches:
            best_match = matches[0]
            logger.info("fuzzy_fallback_matched", query=query, matched=best_match)
            return await self.get_fallback(best_match)

        # Token overlap matching with stop-word filtering
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "of",
            "for", "to", "from", "what", "how", "why", "does", "do", "did", "can",
            "explain", "tell", "me", "about", "show", "describe", "with", "and", "or",
        }
        import re

        q_tokens = set(re.findall(r"\w+", norm_query)) - stop_words
        if q_tokens:
            best_score = 0.0
            best_candidate = None
            for candidate in self._cached_queries:
                c_tokens = set(re.findall(r"\w+", candidate)) - stop_words
                if not c_tokens:
                    continue
                overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
                if overlap > best_score:
                    best_score = overlap
                    best_candidate = candidate

            if best_candidate is not None and best_score >= 0.3:
                logger.info("token_overlap_matched", query=query, matched=best_candidate, score=best_score)
                return await self.get_fallback(best_candidate)

        return None


    async def seed_fallback(self, queries: dict[str, str]) -> None:
        """Seeds the Redis hash with pre-computed query->answer pairs."""
        await self._ensure_connected()
        assert self.redis is not None

        if not queries:
            return

        to_store: dict[str, str] = {}
        for q, a in queries.items():
            norm_q = q.strip().lower()
            sha = hashlib.sha256(norm_q.encode("utf-8")).hexdigest()
            to_store[sha] = a
            to_store[norm_q] = a
            if norm_q not in self._cached_queries:
                self._cached_queries.append(norm_q)

        await self.redis.hset(REDIS_HASH_KEY, mapping=to_store)  # type: ignore
        logger.info("fallback_seeded", count=len(queries))

    async def seed_from_file(self, path: Path | str) -> int:
        """Loads pre-computed Q&A pairs from a JSON file and seeds Redis.

        Supports both:
        - List of {"query": "...", "answer": "..."}
        - Dict of {"query": "answer", ...}
        """
        p = Path(path)
        if not p.exists():
            logger.warning("fallback_file_not_found", path=str(p))
            return 0

        content = json.loads(p.read_text(encoding="utf-8"))
        queries: dict[str, str] = {}

        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "query" in item and "answer" in item:
                    queries[item["query"]] = item["answer"]
        elif isinstance(content, dict):
            if "queries" in content and isinstance(content["queries"], list):
                for item in content["queries"]:
                    if isinstance(item, dict) and "query" in item and "answer" in item:
                        queries[item["query"]] = item["answer"]
            else:
                for k, v in content.items():
                    if isinstance(v, str):
                        queries[k] = v

        await self.seed_fallback(queries)
        return len(queries)

    @property
    def circuit_status(self) -> str:
        """Returns 'closed', 'open', or 'half-open' based on breaker state."""
        return self.breaker.current_state

    async def cached_query_count(self) -> int:
        """Returns number of unique entries in the Redis fallback hash."""
        await self._ensure_connected()
        if not self.redis:
            return 0
        total_keys = await self.redis.hlen(REDIS_HASH_KEY)  # type: ignore
        # Since each query is stored under both sha256 and normalized string,
        # return the number of distinct queries or total keys // 2
        return total_keys // 2 if total_keys > 0 else 0

    async def aclose(self) -> None:
        """Closes Redis connection."""
        if self.redis is not None:
            await self.redis.aclose()  # type: ignore
            self.redis = None
            logger.info("redis_closed")

    async def close(self) -> None:
        """Alias for aclose."""
        await self.aclose()
