"""Phase 1 gate tests for the sensor pipeline.

Run with:  .venv/bin/python -m pytest tests/test_sensor.py -v
All tests must pass before moving to Phase 2.
"""

import time
import pytest
from thermalkit.sensor import SensorBuffer
from thermalkit.sensor.system_info import _read_battery_pct, _read_mem_pressure_gb


# ---------------------------------------------------------------------------
# System info (no sensor hardware required)
# ---------------------------------------------------------------------------

def test_battery_pct_valid():
    pct = _read_battery_pct()
    assert pct is not None, "pmset failed — is this a Mac with a battery?"
    assert 0.0 <= pct <= 100.0, f"Battery % out of range: {pct}"


def test_mem_pressure_valid():
    mem = _read_mem_pressure_gb()
    assert mem is not None, "vm_stat / sysctl failed"
    assert 0.0 < mem < 25.0, f"Memory pressure out of range: {mem} GB"  # M4 Pro = 24 GB


# ---------------------------------------------------------------------------
# Full sensor buffer (requires Swift binary + 10 s wall time)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sensor_buffer_10s():
    """10-second buffer test: no gaps > 1.5 s, all fields in sane ranges."""
    buf = SensorBuffer()
    buf.start()
    try:
        time.sleep(10)
        history = buf.history()
    finally:
        buf.stop()

    assert len(history) >= 8, f"Expected ≥8 samples in 10 s, got {len(history)}"

    # Check for gaps
    timestamps = [r["ts"] for r in history]
    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i - 1]
        assert gap < 1.5, f"Gap of {gap:.2f}s between samples {i-1} and {i}"

    for r in history:
        assert "cpu_temp_c"      in r
        assert "gpu_temp_c"      in r
        assert "batt_pct"        in r
        assert "mem_pressure_gb" in r

        assert 10.0 <= r["cpu_temp_c"] <= 110.0, f"cpu_temp out of range: {r['cpu_temp_c']}"
        assert  0.0 <= r["gpu_temp_c"] <= 110.0, f"gpu_temp out of range: {r['gpu_temp_c']}"
        assert  0.0 <= r["batt_pct"]   <= 100.0, f"batt_pct out of range: {r['batt_pct']}"
        assert  0.0 <  r["mem_pressure_gb"] < 25.0, f"mem_pressure out of range: {r['mem_pressure_gb']}"


@pytest.mark.slow
def test_get_current_state_fast():
    """get_current_state() must return within 5 ms once buffer is populated."""
    buf = SensorBuffer()
    buf.start()
    try:
        # Wait for first sample
        state = buf.get_current_state(timeout=5.0)
        assert state is not None, "No state received within 5 s"

        # Measure subsequent call latency
        start = time.monotonic()
        state2 = buf.get_current_state(timeout=1.0)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert state2 is not None
        assert elapsed_ms < 5.0, f"get_current_state took {elapsed_ms:.1f} ms (limit: 5 ms)"
    finally:
        buf.stop()


@pytest.mark.slow
def test_sensor_buffer_restart():
    """Buffer restarts cleanly: stop() then start() produces fresh samples."""
    buf = SensorBuffer()
    buf.start()
    time.sleep(3)
    buf.stop()

    buf2 = SensorBuffer()
    buf2.start()
    try:
        time.sleep(3)
        state = buf2.get_current_state(timeout=2.0)
        assert state is not None, "No state after restart"
        assert state["cpu_temp_c"] > 0
    finally:
        buf2.stop()
