"""Phase 4 tests for ForgeDaemon."""

import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# ForgeDaemon — unit tests (no model)
# ---------------------------------------------------------------------------

def _make_daemon(tmp_path):
    """Create a ForgeDaemon with mock model/tokenizer for non-inference tests."""
    from thermalkit.daemon.daemon import ForgeDaemon
    from thermalkit.bandit.policy import LinUCBPolicy

    # Minimal stub — we won't call generate() in unit tests
    class _MockModel:
        pass

    class _MockTokenizer:
        pass

    daemon = ForgeDaemon(
        model=_MockModel(),
        tokenizer=_MockTokenizer(),
        policy_path=tmp_path / "policy.npz",
        db_path=tmp_path / "inference.db",
    )
    return daemon


def test_daemon_starts_and_stops(tmp_path):
    daemon = _make_daemon(tmp_path)
    daemon.start()
    assert daemon._running is True
    time.sleep(0.5)
    daemon.stop()
    assert daemon._running is False


def test_daemon_context_manager(tmp_path):
    from thermalkit.daemon.daemon import ForgeDaemon

    class _MockModel:
        pass

    class _MockTokenizer:
        pass

    with ForgeDaemon(
        model=_MockModel(),
        tokenizer=_MockTokenizer(),
        policy_path=tmp_path / "policy.npz",
        db_path=tmp_path / "inference.db",
    ) as daemon:
        assert daemon._running is True
    assert daemon._running is False


def test_daemon_policy_persists_after_stop(tmp_path):
    """Policy file must exist after daemon.stop()."""
    daemon = _make_daemon(tmp_path)
    daemon.start()
    # Manually inject a few updates so call_count > 0
    state = np.ones(6, dtype=np.float32) * 0.5
    daemon._policy.update(state, 0, 50.0)
    daemon._policy.update(state, 1, 80.0)
    daemon.stop()

    policy_path = tmp_path / "policy.npz"
    assert policy_path.exists(), "Policy .npz not saved on stop()"

    from thermalkit.bandit.policy import LinUCBPolicy
    loaded = LinUCBPolicy.load(policy_path)
    assert loaded.call_count == 2


def test_daemon_status_keys(tmp_path):
    daemon = _make_daemon(tmp_path)
    daemon.start()
    time.sleep(1.5)
    st = daemon.status()
    daemon.stop()

    required_keys = {
        "running", "policy_calls", "warmup_done", "last_reward",
        "cpu_temp_c", "batt_pct", "best_action_batch_size", "ucb_scores",
    }
    assert required_keys.issubset(st.keys())
    assert isinstance(st["ucb_scores"], dict)
    assert len(st["ucb_scores"]) == 4  # one per batch_size


def test_daemon_stop_is_idempotent(tmp_path):
    daemon = _make_daemon(tmp_path)
    daemon.start()
    daemon.stop()
    daemon.stop()  # second stop must not raise


# ---------------------------------------------------------------------------
# ForgeDaemon — inference integration (requires MLX model, mark slow)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_daemon_generate_returns_text(tmp_path):
    from thermalkit.daemon.daemon import ForgeDaemon

    MODEL = "mlx-community/Phi-3-mini-4k-instruct-4bit"
    with ForgeDaemon.from_model(
        MODEL,
        policy_path=tmp_path / "policy.npz",
        db_path=tmp_path / "inference.db",
    ) as daemon:
        text = daemon.generate("Say hello in one word.", max_tokens=10)

    assert isinstance(text, str) and len(text) > 0


@pytest.mark.slow
def test_daemon_policy_updates_over_calls(tmp_path):
    """Policy call_count must increase across multiple generate() calls."""
    from thermalkit.daemon.daemon import ForgeDaemon

    MODEL = "mlx-community/Phi-3-mini-4k-instruct-4bit"
    prompts = [
        "Name a color.",
        "What is 2+2?",
        "Say hi.",
        "Name a planet.",
        "What is Python?",
    ]

    with ForgeDaemon.from_model(
        MODEL,
        policy_path=tmp_path / "policy.npz",
        db_path=tmp_path / "inference.db",
    ) as daemon:
        for p in prompts:
            daemon.generate(p, max_tokens=15)
        final_count = daemon.call_count

    # After warmup (4 calls), policy starts updating — by call 5 we should have ≥ 1 update
    assert final_count >= 1, f"Expected policy_calls >= 1, got {final_count}"


@pytest.mark.slow
def test_daemon_policy_persists_and_reloads(tmp_path):
    """After a session, loading a new daemon picks up the saved policy."""
    from thermalkit.daemon.daemon import ForgeDaemon

    MODEL = "mlx-community/Phi-3-mini-4k-instruct-4bit"
    policy_path = tmp_path / "policy.npz"
    db_path = tmp_path / "inference.db"

    # Session 1
    with ForgeDaemon.from_model(MODEL, policy_path=policy_path, db_path=db_path) as d1:
        for _ in range(6):
            d1.generate("Hello.", max_tokens=10)
        count_after_s1 = d1.call_count

    # Session 2 — must load the saved policy
    with ForgeDaemon.from_model(MODEL, policy_path=policy_path, db_path=db_path) as d2:
        count_at_start_s2 = d2.call_count

    assert count_at_start_s2 == count_after_s1, (
        f"Session 2 started with {count_at_start_s2} calls, "
        f"expected {count_after_s1} from session 1"
    )


@pytest.mark.slow
def test_daemon_log_grows_with_calls(tmp_path):
    """SQLite db must have one row per generate() call."""
    from thermalkit.daemon.daemon import ForgeDaemon

    MODEL = "mlx-community/Phi-3-mini-4k-instruct-4bit"
    db_path = tmp_path / "inference.db"
    n_calls = 4

    with ForgeDaemon.from_model(
        MODEL,
        policy_path=tmp_path / "policy.npz",
        db_path=db_path,
    ) as daemon:
        for _ in range(n_calls):
            daemon.generate("Hello.", max_tokens=10)
        daemon._log.flush(timeout=5.0)

    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    conn.close()
    assert count == n_calls
