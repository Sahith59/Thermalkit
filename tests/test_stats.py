"""Feature 2 tests — thermalkit stats command."""

import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest


def run_stats(db_path=None, extra_env=None):
    """Run thermalkit stats and return (returncode, stdout+stderr)."""
    import os
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent)}
    if db_path:
        # Patch by injecting a custom HOME so the stats command finds our test db
        env["THERMALKIT_TEST_DB"] = str(db_path)
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(
        [sys.executable, "-m", "thermalkit.cli", "stats"],
        capture_output=True, text=True, timeout=15, env=env,
    )
    return result.returncode, result.stdout + result.stderr


def _make_db(path: Path, n_rows: int = 20) -> Path:
    """Create a minimal inference.db with synthetic data."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, cpu_temp REAL, gpu_temp REAL, power_w REAL,
            mem_pressure REAL, batt_pct REAL, compute_unit TEXT,
            batch_size INTEGER, tokens_generated INTEGER,
            tok_sec REAL, reward REAL
        )
    """)
    base_ts = time.time() - n_rows * 2
    for i in range(n_rows):
        conn.execute(
            "INSERT INTO calls (ts, cpu_temp, gpu_temp, batt_pct, mem_pressure, "
            "compute_unit, batch_size, tokens_generated, tok_sec, reward) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                base_ts + i * 2,
                50.0 + i * 0.5,          # cpu_temp climbs from 50 → 60°C
                0.0,
                80.0,
                4.0,
                "gpu",
                1 if i % 3 != 0 else 2,  # mostly batch_size=1
                150,
                100.0 + i * 0.2,          # tok/sec improves slightly
                None,
            ),
        )
    conn.commit()
    conn.close()
    return path


# ---------------------------------------------------------------------------
# Tests against real ~/.thermalkit/inference.db (skipped if missing)
# ---------------------------------------------------------------------------

def test_stats_runs_without_crash():
    rc, out = run_stats()
    # Should either show dashboard or "No inference log found" — never crash
    assert rc == 0


def test_stats_no_db_shows_helpful_message(tmp_path, monkeypatch):
    """When no DB exists, stats shows a clear message."""
    monkeypatch.setenv("HOME", str(tmp_path))
    rc, out = run_stats()
    assert "No inference log" in out or "benchmark" in out.lower()
    assert rc == 0


# ---------------------------------------------------------------------------
# Tests with a synthetic DB
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_db(tmp_path):
    db = tmp_path / "inference.db"
    _make_db(db, n_rows=20)
    return db


def _run_with_real_db():
    """Check if real DB exists."""
    real = Path.home() / ".thermalkit" / "inference.db"
    old  = Path.home() / ".forge" / "inference.db"
    return real.exists() or old.exists()


@pytest.mark.skipif(not _run_with_real_db(), reason="No real inference.db found")
def test_stats_shows_dashboard_sections():
    rc, out = run_stats()
    assert "ThermalKit Learning Dashboard" in out
    assert "Throughput" in out
    assert "Thermal range" in out
    assert "Bandit" in out


@pytest.mark.skipif(not _run_with_real_db(), reason="No real inference.db found")
def test_stats_shows_tok_sec():
    rc, out = run_stats()
    assert "tok/sec" in out or "tok/s" in out


@pytest.mark.skipif(not _run_with_real_db(), reason="No real inference.db found")
def test_stats_shows_date_range():
    rc, out = run_stats()
    assert "202" in out  # year in date range


@pytest.mark.skipif(not _run_with_real_db(), reason="No real inference.db found")
def test_stats_shows_bandit_section():
    rc, out = run_stats()
    assert "batch_size" in out or "Preferred" in out


@pytest.mark.skipif(not _run_with_real_db(), reason="No real inference.db found")
def test_stats_completes_fast():
    import time as _time
    t0 = _time.perf_counter()
    run_stats()
    elapsed = _time.perf_counter() - t0
    assert elapsed < 5.0, f"stats took {elapsed:.1f}s"
