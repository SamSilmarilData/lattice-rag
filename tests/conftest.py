import os

import pytest
from dotenv import load_dotenv

# Automatically load .env environment variables for all test suites if present
load_dotenv()

# Provide hermetic dummy fallback keys for CI / test environments when .env is absent
os.environ.setdefault("TYPESAFE_API_KEY", "mock-typesafe-key-for-hermetic-unit-tests")
os.environ.setdefault("GROQ_API_KEY", "mock-groq-key-for-hermetic-unit-tests")
os.environ.setdefault("GEMINI_API_KEY", "mock-gemini-key-for-hermetic-unit-tests")


@pytest.fixture(autouse=True)
def reset_config_cache():
    """Ensure AppConfig cache is fresh for each test and cleared afterwards."""
    from lattice_rag.config import get_config

    get_config.cache_clear()
    yield
    get_config.cache_clear()
