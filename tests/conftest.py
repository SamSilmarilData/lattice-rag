import pytest
from dotenv import load_dotenv

# Automatically load .env environment variables for all test suites
load_dotenv()


@pytest.fixture(autouse=True, scope="session")
def cleanup_process_pool():
    yield
    try:
        from lattice_rag.orchestration.pool import shutdown_pool
        shutdown_pool()
    except Exception:
        pass
