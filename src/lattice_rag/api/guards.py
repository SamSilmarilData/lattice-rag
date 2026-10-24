"""Route guards protecting write and cost-incurring endpoints."""

from __future__ import annotations

import hmac
import logging
import time
from collections.abc import Callable
from typing import Any

from litestar.connection import ASGIConnection
from litestar.exceptions import NotAuthorizedException, TooManyRequestsException
from litestar.handlers.base import BaseRouteHandler

from lattice_rag.api.client_identity import client_address

logger = logging.getLogger(__name__)

ADMIN_KEY_HEADER = "X-Admin-Key"
ADMIN_KEY_THROTTLE_STATE = "admin_key_throttle"


class AdminKeyThrottle:
    """Per-client lockout after repeated failed admin-key attempts.

    A client that fails ``max_failures`` times within ``window_seconds`` of its first failure
    is locked out until that window ends; while locked, its key is not even checked. A
    successful attempt clears the client's failures. State is in-process, so each server
    worker keeps its own counts.
    """

    def __init__(
        self,
        max_failures: int = 5,
        window_seconds: float = 300.0,
        max_tracked_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self.max_tracked_clients = max_tracked_clients
        self._clock = clock
        # client -> (failure count, time of first failure in the window)
        self._failures: dict[str, tuple[int, float]] = {}

    @property
    def tracked_clients(self) -> int:
        return len(self._failures)

    def _live_entry(self, client: str) -> tuple[int, float] | None:
        entry = self._failures.get(client)
        if entry is not None and self._clock() - entry[1] >= self.window_seconds:
            del self._failures[client]
            return None
        return entry

    def is_locked(self, client: str) -> bool:
        entry = self._live_entry(client)
        return entry is not None and entry[0] >= self.max_failures

    def record_failure(self, client: str) -> None:
        entry = self._live_entry(client)
        if entry is None:
            self._make_room()
            entry = (0, self._clock())
        count = entry[0] + 1
        self._failures[client] = (count, entry[1])
        if count == self.max_failures:
            logger.warning("admin_key_lockout client=%s window_s=%s", client, self.window_seconds)

    def record_success(self, client: str) -> None:
        self._failures.pop(client, None)

    def _make_room(self) -> None:
        if len(self._failures) < self.max_tracked_clients:
            return
        now = self._clock()
        for client, (_, started) in list(self._failures.items()):
            if now - started >= self.window_seconds:
                del self._failures[client]
        while len(self._failures) >= self.max_tracked_clients:
            # Evict the client whose window started earliest (dicts keep insertion order)
            del self._failures[next(iter(self._failures))]


def _throttle_for(connection: ASGIConnection[Any, Any, Any, Any]) -> AdminKeyThrottle:
    state = connection.app.state
    throttle = state.get(ADMIN_KEY_THROTTLE_STATE)
    if not isinstance(throttle, AdminKeyThrottle):
        throttle = AdminKeyThrottle()
        state[ADMIN_KEY_THROTTLE_STATE] = throttle
    return throttle


def require_admin_key(connection: ASGIConnection[Any, Any, Any, Any], _: BaseRouteHandler) -> None:
    """Allow the request only when X-Admin-Key matches the configured ADMIN_API_KEY.

    Fails closed: when no admin key is configured, every guarded request is rejected.
    Clients that repeatedly send a wrong key are locked out with 429 (see AdminKeyThrottle).
    """
    config = getattr(connection.app.state, "config", None)
    throttle = _throttle_for(connection)
    client = client_address(connection, getattr(config, "trusted_proxy_hops", 0))
    if throttle.is_locked(client):
        raise TooManyRequestsException(
            detail="Too many failed X-Admin-Key attempts. Try again later."
        )

    admin_key = getattr(config, "admin_api_key", None)
    expected = admin_key.get_secret_value() if admin_key else ""
    provided = connection.headers.get(ADMIN_KEY_HEADER, "")

    if not expected or not hmac.compare_digest(provided.encode(), expected.encode()):
        throttle.record_failure(client)
        raise NotAuthorizedException(detail="A valid X-Admin-Key header is required.")
    throttle.record_success(client)
