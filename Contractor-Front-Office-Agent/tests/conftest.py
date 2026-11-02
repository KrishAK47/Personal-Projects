"""Shared fixtures: every test gets its own copy of the sample data and a frozen clock."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")  # lets `pytest -m eval` pick up ANTHROPIC_API_KEY locally

import data_store as ds  # noqa: E402
import seed_data  # noqa: E402
import store  # noqa: E402

# A Tuesday morning, before the first slot of the day.
FROZEN_NOW = datetime(2026, 10, 20, 8, 0)


@pytest.fixture
def clock(monkeypatch):
    """Freeze ds.now(). Tests can move time with clock.set(datetime(...))."""

    class Clock:
        current = FROZEN_NOW

        def set(self, when: datetime) -> None:
            self.current = when

    c = Clock()
    monkeypatch.setattr(ds, "now", lambda: c.current)
    return c


@pytest.fixture
def data_dir(tmp_path, monkeypatch, clock):
    """Use CSV storage in a temp folder seeded with fresh sample data (never the real database)."""
    monkeypatch.setattr(store, "get_store", lambda: store.CsvStore())
    monkeypatch.setattr(seed_data, "_checked_on", "")
    paths = {
        "SCHEDULE_CSV": tmp_path / "schedule.csv",
        "JOBS_CSV": tmp_path / "jobs.csv",
        "CUSTOMERS_CSV": tmp_path / "customers.csv",
    }
    monkeypatch.setattr(ds, "DATA_DIR", tmp_path)
    for name, path in paths.items():
        monkeypatch.setattr(ds, name, path)
    monkeypatch.setattr(ds, "_FILES", {
        paths["SCHEDULE_CSV"]: ds.SCHEDULE_COLUMNS,
        paths["JOBS_CSV"]: ds.JOBS_COLUMNS,
        paths["CUSTOMERS_CSV"]: ds.CUSTOMERS_COLUMNS,
    })
    seed_data.seed(reset=True)
    return tmp_path
