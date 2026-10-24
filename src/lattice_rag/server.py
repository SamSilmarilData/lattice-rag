from __future__ import annotations

import logging

from granian import Granian
from granian.constants import Interfaces

from lattice_rag.config import load_config

logger = logging.getLogger(__name__)


def main(host: str | None = None, port: int | None = None) -> None:
    """Start the high-throughput Granian Rust ASGI server runner.

    Explicit host/port (from the CLI) win over HOST/PORT from the environment. A missing
    TYPESAFE_API_KEY does not stop the server: the app boots in degraded hermetic mode.
    """
    config = load_config(require_typesafe=False)
    bind_host = host or config.host
    bind_port = port or config.port

    print(f"🚀 Starting lattice-rag Granian ASGI server on http://{bind_host}:{bind_port}")
    server = Granian(
        "lattice_rag.app:create_app",
        address=bind_host,
        port=bind_port,
        interface=Interfaces.ASGI,
        factory=True,
        workers=1,
        runtime_threads=2,
    )
    server.serve()


if __name__ == "__main__":
    main()
