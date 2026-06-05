import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# TokSecMeter
# ---------------------------------------------------------------------------

def test_tok_sec_meter_instantaneous():
    from thermalkit.mlx_wrap.tok_sec_meter import TokSecMeter
    m = TokSecMeter()
    tps = m.record(200, 4.0)  # 50 tok/s
    assert abs(tps - 50.0) < 0.01


def test_tok_sec_meter_reward_needs_10s():
    from thermalkit.mlx_wrap.tok_sec_meter import TokSecMeter
    m = TokSecMeter()
    m.record(100, 2.0)  # single entry — span < 10s
    assert m.get_reward() is None


def test_tok_sec_meter_window():
    from thermalkit.mlx_wrap.tok_sec_meter import TokSecMeter
    import time
    m = TokSecMeter()
    # Inject entries spanning > 10s by backdating internal state
    now = time.time()
    m._entries.append((now - 12.0, 40.0))
    m._entries.append((now - 6.0, 60.0))
    m._entries.append((now, 50.0))
    reward = m.get_reward()
    assert reward is not None
    assert abs(reward - 50.0) < 1.0  # mean of 40, 60, 50


# ---------------------------------------------------------------------------
# StateBuilder
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_state_builder_shape_and_range():
    from thermalkit.sensor import SensorBuffer
    from thermalkit.mlx_wrap.state_builder import StateBuilder, FEATURE_DIM
    buf = SensorBuffer()
    buf.start()
    time.sleep(2)
    sb = StateBuilder(buf)
    vec = sb.get()
    buf.stop()
    assert vec.shape == (FEATURE_DIM,)
    assert vec.dtype == np.float32
    assert (vec >= 0.0).all() and (vec <= 1.0).all()


def test_state_builder_defaults():
    """StateBuilder works even with a buffer that never started (uses fallbacks)."""
    from thermalkit.sensor import SensorBuffer
    from thermalkit.mlx_wrap.state_builder import StateBuilder, FEATURE_DIM
    buf = SensorBuffer()  # never started
    sb = StateBuilder(buf)
    vec = sb.get()
    assert vec.shape == (FEATURE_DIM,)
    assert (vec >= 0.0).all() and (vec <= 1.0).all()


# ---------------------------------------------------------------------------
# InferenceLog
# ---------------------------------------------------------------------------

def test_inference_log_schema(tmp_path):
    from thermalkit.mlx_wrap.inference_log import InferenceLog
    db = tmp_path / "test.db"
    log = InferenceLog(db_path=db)
    log.start()
    log.log(
        {
            "ts": 1_700_000_000.0,
            "cpu_temp": 55.0,
            "gpu_temp": 45.0,
            "power_w": 10.0,
            "mem_pressure": 4.0,
            "batt_pct": 80.0,
            "compute_unit": "gpu",
            "batch_size": 1,
            "tokens_generated": 100,
            "tok_sec": 50.0,
            "reward": None,
        }
    )
    log.flush(timeout=3.0)
    conn = sqlite3.connect(str(db))
    cols = [row[1] for row in conn.execute("PRAGMA table_info(calls)").fetchall()]
    rows = conn.execute("SELECT * FROM calls").fetchall()
    conn.close()

    expected_cols = [
        "id", "ts", "cpu_temp", "gpu_temp", "power_w", "mem_pressure", "batt_pct",
        "compute_unit", "batch_size", "tokens_generated", "tok_sec", "reward",
    ]
    assert cols == expected_cols
    assert len(rows) == 1
    row = rows[0]
    assert row[cols.index("compute_unit")] == "gpu"
    assert row[cols.index("tokens_generated")] == 100
    assert abs(row[cols.index("tok_sec")] - 50.0) < 0.01


def test_inference_log_multiple_rows(tmp_path):
    from thermalkit.mlx_wrap.inference_log import InferenceLog
    db = tmp_path / "multi.db"
    log = InferenceLog(db_path=db)
    log.start()
    for i in range(5):
        log.log({"ts": float(i), "compute_unit": "cpu", "batch_size": 1,
                 "tokens_generated": 50, "tok_sec": 25.0})
    log.flush(timeout=3.0)
    conn = sqlite3.connect(str(db))
    count = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    conn.close()
    assert count == 5


# ---------------------------------------------------------------------------
# thermalkit generate() — output correctness (requires MLX model, mark slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_generate_output_matches_mlx_lm(tmp_path):
    """thermalkit.generate() must produce non-empty output matching mlx_lm.generate()."""
    import mlx_lm
    import thermalkit.mlx_wrap._generate as fgen

    MODEL = "mlx-community/Phi-3-mini-4k-instruct-4bit"
    model, tokenizer = mlx_lm.load(MODEL)
    prompt = "What is 2 + 2? Answer in one word."

    # Reset thermalkit module-level state so log goes to tmp_path
    from thermalkit.mlx_wrap.inference_log import InferenceLog
    fgen._log = InferenceLog(db_path=tmp_path / "gen_test.db")
    fgen._log.start()
    fgen._sensor_buf = None  # let it lazy-init

    expected = mlx_lm.generate(model, tokenizer, prompt, max_tokens=20)
    actual = fgen.generate(model, tokenizer, prompt, max_tokens=20)

    # Both should be non-empty strings
    assert isinstance(actual, str) and len(actual) > 0
    assert isinstance(expected, str) and len(expected) > 0


@pytest.mark.slow
def test_generate_logs_to_db(tmp_path):
    """After thermalkit.generate(), SQLite db must have one row with correct fields."""
    import mlx_lm
    import thermalkit.mlx_wrap._generate as fgen
    from thermalkit.mlx_wrap.inference_log import InferenceLog

    MODEL = "mlx-community/Phi-3-mini-4k-instruct-4bit"
    model, tokenizer = mlx_lm.load(MODEL)

    db_path = tmp_path / "log_test.db"
    fgen._log = InferenceLog(db_path=db_path)
    fgen._log.start()
    fgen._sensor_buf = None

    fgen.generate(model, tokenizer, "Say hello.", max_tokens=10)
    fgen._log.flush(timeout=5.0)

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT * FROM calls").fetchall()
    cols = [row[1] for row in conn.execute("PRAGMA table_info(calls)").fetchall()]
    conn.close()

    assert len(rows) == 1
    row = rows[0]
    assert row[cols.index("compute_unit")] == "gpu"
    assert row[cols.index("tokens_generated")] > 0
    assert row[cols.index("tok_sec")] > 0
