"""Feature 5 tests — thermalkit export command."""

import csv
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest


HEADERS = [
    "ts", "cpu_temp_c", "gpu_temp_c", "mem_pressure_gb", "batt_pct",
    "compute_unit", "batch_size", "tokens_generated", "tok_sec", "reward",
]


def _make_db(path: Path, n_rows: int = 10) -> Path:
    """Create a minimal inference.db for testing."""
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
            "INSERT INTO calls (ts, cpu_temp, gpu_temp, mem_pressure, batt_pct, "
            "compute_unit, batch_size, tokens_generated, tok_sec, reward) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                base_ts + i * 2,
                50.0 + i,
                0.0,
                4.0 + i * 0.1,
                80.0,
                "gpu",
                1 if i % 2 == 0 else 2,
                150,
                100.0 + i,
                95.0 + i if i >= 5 else None,  # reward is NULL for first 5
            ),
        )
    conn.commit()
    conn.close()
    return path


def _run_export(tmp_db: Path, *extra_args):
    """Run thermalkit export with HOME pointing at a temp dir that contains the db."""
    import tempfile, shutil
    # Create a fake home with ~/.thermalkit/inference.db
    fake_home = Path(tempfile.mkdtemp())
    tk_dir = fake_home / ".thermalkit"
    tk_dir.mkdir()
    shutil.copy(tmp_db, tk_dir / "inference.db")

    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).parent.parent),
        "HOME": str(fake_home),
    }
    result = subprocess.run(
        [sys.executable, "-m", "thermalkit.cli", "export", *extra_args],
        capture_output=True, text=True, timeout=15, env=env,
        cwd=str(fake_home),  # output files land here
    )
    return result.returncode, result.stdout + result.stderr, fake_home


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def test_csv_export_creates_file(tmp_path):
    db = _make_db(tmp_path / "inference.db")
    rc, out, home = _run_export(db, "--output", str(tmp_path / "out.csv"), "--format", "csv")
    assert rc == 0, f"export failed:\n{out}"
    assert (tmp_path / "out.csv").exists()


def test_csv_export_correct_row_count(tmp_path):
    db = _make_db(tmp_path / "inference.db", n_rows=10)
    out_path = str(tmp_path / "out.csv")
    rc, out, home = _run_export(db, "--output", out_path, "--format", "csv")
    assert rc == 0
    with open(out_path) as f:
        reader = csv.reader(f)
        rows = list(reader)
    assert len(rows) == 11  # 1 header + 10 data rows


def test_csv_export_correct_headers(tmp_path):
    db = _make_db(tmp_path / "inference.db")
    out_path = str(tmp_path / "out.csv")
    rc, out, home = _run_export(db, "--output", out_path)
    with open(out_path) as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header == HEADERS


def test_csv_export_null_becomes_empty_string(tmp_path):
    """NULL values in DB must export as empty strings, not 'None'."""
    db = _make_db(tmp_path / "inference.db", n_rows=3)
    out_path = str(tmp_path / "out.csv")
    rc, out, home = _run_export(db, "--output", out_path)
    with open(out_path) as f:
        content = f.read()
    assert "None" not in content


def test_csv_export_shows_row_count_in_output(tmp_path):
    db = _make_db(tmp_path / "inference.db", n_rows=7)
    rc, out, home = _run_export(db, "--output", str(tmp_path / "out.csv"))
    assert "7" in out


def test_csv_export_shows_columns_in_output(tmp_path):
    db = _make_db(tmp_path / "inference.db")
    rc, out, home = _run_export(db, "--output", str(tmp_path / "out.csv"))
    assert "tok_sec" in out
    assert "batch_size" in out


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def test_json_export_creates_file(tmp_path):
    db = _make_db(tmp_path / "inference.db")
    out_path = str(tmp_path / "out.json")
    rc, out, home = _run_export(db, "--output", out_path, "--format", "json")
    assert rc == 0
    assert Path(out_path).exists()


def test_json_export_valid_structure(tmp_path):
    db = _make_db(tmp_path / "inference.db", n_rows=5)
    out_path = str(tmp_path / "out.json")
    rc, out, home = _run_export(db, "--output", out_path, "--format", "json")
    with open(out_path) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 5
    assert all(isinstance(row, dict) for row in data)


def test_json_export_correct_keys(tmp_path):
    db = _make_db(tmp_path / "inference.db", n_rows=3)
    out_path = str(tmp_path / "out.json")
    rc, out, home = _run_export(db, "--output", out_path, "--format", "json")
    with open(out_path) as f:
        data = json.load(f)
    assert set(data[0].keys()) == set(HEADERS)


# ---------------------------------------------------------------------------
# --since filter
# ---------------------------------------------------------------------------

def test_since_filter_reduces_row_count(tmp_path):
    """--since tomorrow should return 0 rows."""
    from datetime import datetime, timedelta
    db = _make_db(tmp_path / "inference.db", n_rows=10)
    out_path = str(tmp_path / "out.csv")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    rc, out, home = _run_export(db, "--output", out_path, "--since", tomorrow)
    assert rc == 0
    with open(out_path) as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1  # header only


def test_since_filter_past_date_returns_all(tmp_path):
    """--since a past date should return all rows."""
    db = _make_db(tmp_path / "inference.db", n_rows=8)
    out_path = str(tmp_path / "out.csv")
    rc, out, home = _run_export(db, "--output", out_path, "--since", "2020-01-01")
    assert rc == 0
    with open(out_path) as f:
        rows = list(csv.reader(f))
    assert len(rows) == 9  # header + 8 rows


def test_since_shows_filter_in_output(tmp_path):
    db = _make_db(tmp_path / "inference.db")
    rc, out, home = _run_export(
        db, "--output", str(tmp_path / "out.csv"), "--since", "2026-01-01"
    )
    assert "2026-01-01" in out


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_invalid_format_exits_nonzero(tmp_path):
    db = _make_db(tmp_path / "inference.db")
    rc, out, home = _run_export(db, "--output", str(tmp_path / "out.xyz"), "--format", "xlsx")
    assert rc != 0


def test_invalid_since_date_exits_nonzero(tmp_path):
    db = _make_db(tmp_path / "inference.db")
    rc, out, home = _run_export(db, "--output", str(tmp_path / "out.csv"), "--since", "not-a-date")
    assert rc != 0


def test_no_db_exits_nonzero(tmp_path):
    """When no DB exists at all, export should exit 1."""
    import tempfile
    fake_home = Path(tempfile.mkdtemp())  # empty home, no .thermalkit dir
    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).parent.parent),
        "HOME": str(fake_home),
    }
    result = subprocess.run(
        [sys.executable, "-m", "thermalkit.cli", "export",
         "--output", str(tmp_path / "out.csv")],
        capture_output=True, text=True, timeout=10, env=env,
    )
    assert result.returncode != 0
