"""Trustworthy client identity for rate limiting and admin-key throttling.

By default the socket peer identifies the client and X-Forwarded-For is ignored, since any
client can send that header. When ``TRUSTED_PROXY_HOPS`` is N > 0, each of the N reverse
proxies in front of the app appends the address it received the request from, so the
client is the Nth entry from the right. Entries further left are client-supplied and ignored.
"""

from __future__ import annotations

from typing import Any

from litestar.connection import ASGIConnection

FORWARDED_FOR_HEADER = "x-forwarded-for"
UNKNOWN_CLIENT = "unknown"


def resolve_client_address(
    peer: str | None, forwarded_for: str | None, trusted_proxy_hops: int
) -> str:
    """Return the client address given the socket peer and the X-Forwarded-For header.

    A header with fewer entries than trusted hops cannot have been written by every proxy,
    so it is not trusted and the socket peer is used instead.
    """
    fallback = peer or UNKNOWN_CLIENT
    if trusted_proxy_hops <= 0 or not forwarded_for:
        return fallback
    entries = [part.strip() for part in forwarded_for.split(",") if part.strip()]
    if len(entries) < trusted_proxy_hops:
        return fallback
    return entries[-trusted_proxy_hops]


def client_address(connection: ASGIConnection[Any, Any, Any, Any], trusted_proxy_hops: int) -> str:
    """Resolve the client address of a request or websocket connection."""
    peer = connection.client.host if connection.client else None
    return resolve_client_address(
        peer, connection.headers.get(FORWARDED_FOR_HEADER), trusted_proxy_hops
    )
