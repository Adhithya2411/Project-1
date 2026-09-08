"""Shared pytest fixtures.

The engine is session-scoped over a temporary database: seeding and index
building take a second or two, and every test wants the same seeded corpus.
Tests that mutate the corpus use the ``fresh_engine`` fixture instead.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from ahrag.config import RouterConfig, Settings
from ahrag.db import Database
from ahrag.pipeline import AHRAGEngine

# Fixed reference date so freshness-dependent assertions do not start failing
# when the real clock passes a document's effective date.
TEST_TODAY = date(2026, 8, 19)

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Settings pointing at a temporary data directory."""
    data_dir = tmp_path_factory.mktemp("ahrag-data")
    return Settings(
        data_dir=data_dir,
        db_path=data_dir / "test.sqlite3",
        router_config=REPO_ROOT / "config" / "router.yaml",
        verbose_audit=False,
    )


@pytest.fixture(scope="session")
def config(settings: Settings) -> RouterConfig:
    """The routing policy loaded from the repository config."""
    return RouterConfig.load(settings.router_config)


@pytest.fixture(scope="session")
def engine(settings: Settings, config: RouterConfig) -> AHRAGEngine:
    """A seeded engine shared across read-only tests."""
    db = Database(settings.db_path)
    built = AHRAGEngine(settings=settings, config=config, db=db, today=TEST_TODAY)
    result = built.seed(reset=True)
    assert result.documents > 0, f"seeding failed: {result.errors}"
    assert not result.errors, result.errors
    return built


@pytest.fixture
def fresh_engine(
    tmp_path: Path, config: RouterConfig
) -> AHRAGEngine:
    """A seeded engine on its own database, for tests that mutate the corpus."""
    local_settings = Settings(
        data_dir=tmp_path,
        db_path=tmp_path / "fresh.sqlite3",
        router_config=REPO_ROOT / "config" / "router.yaml",
    )
    db = Database(local_settings.db_path)
    built = AHRAGEngine(
        settings=local_settings, config=config, db=db, today=TEST_TODAY
    )
    built.seed(reset=True)
    return built
