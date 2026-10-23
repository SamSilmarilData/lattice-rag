"""lattice-rag: In-process, zero-cloud-cost Hybrid GraphRAG engine & CI/CD evaluation suite."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lattice-rag")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+unknown"
