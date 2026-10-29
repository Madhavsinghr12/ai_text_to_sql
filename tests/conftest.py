from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from text2sql.seed import build_database  # noqa: E402


@pytest.fixture(scope="session")
def demo_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A throwaway copy of the demo database, built once per test session."""
    path = tmp_path_factory.mktemp("db") / "retail.db"
    return build_database(path)
