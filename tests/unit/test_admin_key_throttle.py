"""Repeated failed admin-key attempts from one client are throttled."""

from __future__ import annotations

import pytest
from litestar import Litestar, post
from litestar.datastructures import State
from litestar.testing import AsyncTestClient

from lattice_rag.api.guards import ADMIN_KEY_HEADER, AdminKeyThrottle, require_admin_key
from lattice_rag.config import AppConfig
from lattice_rag.security import SecretStr

ADMIN_KEY = "admin-secret"
ATTACKER = "203.0.113.66"
OPERATOR = "198.51.100.7"


@post("/protected", guards=[require_admin_key])
async def protected() -> str:
    return "ok"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _app(throttle: AdminKeyThrottle | None = None) -> Litestar:
    # One trusted hop lets the tests present distinct clients via X-Forwarded-For
    config = AppConfig(
        typesafe_api_key=SecretStr("ts"),
        groq_api_key=SecretStr(""),
        gemini_api_key=SecretStr(""),
        admin_api_key=SecretStr(ADMIN_KEY),
        trusted_proxy_hops=1,
    )
    state: dict[str, object] = {"config": config}
    if throttle is not None:
        state["admin_key_throttle"] = throttle
    return Litestar(route_handlers=[protected], state=State(state))


async def _attempt(client: AsyncTestClient, ip: str, key: str) -> int:
    resp = await client.post("/protected", headers={ADMIN_KEY_HEADER: key, "X-Forwarded-For": ip})
    return resp.status_code


@pytest.mark.asyncio
async def test_repeated_bad_keys_are_throttled_even_when_a_valid_key_follows():
    async with AsyncTestClient(app=_app(AdminKeyThrottle(max_failures=3))) as client:
        assert [await _attempt(client, ATTACKER, "guess") for _ in range(3)] == [401, 401, 401]
        assert await _attempt(client, ATTACKER, "guess") == 429
        # The key is not even checked while locked out, so guessing it right does not help
        assert await _attempt(client, ATTACKER, ADMIN_KEY) == 429


@pytest.mark.asyncio
async def test_valid_key_is_not_blocked_by_another_clients_failures():
    async with AsyncTestClient(app=_app(AdminKeyThrottle(max_failures=3))) as client:
        for _ in range(5):
            await _attempt(client, ATTACKER, "guess")
        assert await _attempt(client, OPERATOR, ADMIN_KEY) == 201


@pytest.mark.asyncio
async def test_lockout_expires_after_the_window():
    clock = FakeClock()
    throttle = AdminKeyThrottle(max_failures=2, window_seconds=60, clock=clock)
    async with AsyncTestClient(app=_app(throttle)) as client:
        for _ in range(2):
            await _attempt(client, ATTACKER, "guess")
        assert await _attempt(client, ATTACKER, ADMIN_KEY) == 429
        clock.now += 61
        assert await _attempt(client, ATTACKER, ADMIN_KEY) == 201


@pytest.mark.asyncio
async def test_successful_attempt_clears_earlier_failures():
    async with AsyncTestClient(app=_app(AdminKeyThrottle(max_failures=2))) as client:
        assert await _attempt(client, OPERATOR, "typo") == 401
        assert await _attempt(client, OPERATOR, ADMIN_KEY) == 201
        assert await _attempt(client, OPERATOR, "typo") == 401
        assert await _attempt(client, OPERATOR, ADMIN_KEY) == 201


@pytest.mark.asyncio
async def test_guard_throttles_with_default_throttle_when_none_is_configured():
    async with AsyncTestClient(app=_app()) as client:
        statuses = [await _attempt(client, ATTACKER, "guess") for _ in range(20)]
        assert statuses[0] == 401
        assert statuses[-1] == 429


def test_tracked_clients_are_bounded():
    throttle = AdminKeyThrottle(max_failures=1, max_tracked_clients=2)
    for ip in ("10.0.0.1", "10.0.0.2", "10.0.0.3"):
        throttle.record_failure(ip)
    assert throttle.tracked_clients == 2
    assert throttle.is_locked("10.0.0.3")
