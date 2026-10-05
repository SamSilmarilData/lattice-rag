from __future__ import annotations

import logging
from granian import Granian

from lattice_rag.config import get_config

logger = logging.getLogger(__name__)


def main() -> None:
    """Start the high-throughput Granian Rust ASGI server runner."""
    try:
        config = get_config()
        host = config.host
        port = config.port
    except Exception as e:
        logger.warning("Could not read configuration, falling back to 0.0.0.0:8000: %s", e)
        host = "0.0.0.0"
        port = 8000

    print(f"🚀 Starting lattice-rag Granian ASGI server on http://{host}:{port}")
    server = Granian(
        "lattice_rag.app:create_app",
        address=host,
        port=port,
        interface="asgi",
        factory=True,
        workers=1,
        threads=4,
    )
    server.serve()


if __name__ == "__main__":
    main()
