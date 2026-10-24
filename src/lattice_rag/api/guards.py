"""Route guards protecting write and cost-incurring endpoints."""

from __future__ import annotations

import hmac
from typing import Any

from litestar.connection import ASGIConnection
from litestar.exceptions import NotAuthorizedException
from litestar.handlers.base import BaseRouteHandler

ADMIN_KEY_HEADER = "X-Admin-Key"


def require_admin_key(connection: ASGIConnection[Any, Any, Any, Any], _: BaseRouteHandler) -> None:
    """Allow the request only when X-Admin-Key matches the configured ADMIN_API_KEY.

    Fails closed: when no admin key is configured, every guarded request is rejected.
    """
    config = getattr(connection.app.state, "config", None)
    admin_key = getattr(config, "admin_api_key", None)
    expected = admin_key.get_secret_value() if admin_key else ""
    provided = connection.headers.get(ADMIN_KEY_HEADER, "")

    if not expected or not hmac.compare_digest(provided.encode(), expected.encode()):
        raise NotAuthorizedException(detail="A valid X-Admin-Key header is required.")
