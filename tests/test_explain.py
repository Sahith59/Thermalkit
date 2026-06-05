"""Feature 4 tests — --explain flag and decision trace."""

import io
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from thermalkit.bandit.policy import LinUCBPolicy
from thermalkit.bandit.action_space import N_ACTIONS, batch_size_for
from thermalkit.daemon.daemon import _print_explain_pre, _print_explain_post, _SEP


# ---------------------------------------------------------------------------
# StateBuilder.get_with_raw
# ---------------------------------------------------------------------------

def test_get_with_raw_returns_tuple():
    from thermalkit.mlx_wrap.state_builder import StateBuilder
    from thermalkit.sensor import SensorBuffer

    buf = SensorBuffer()  # never started — uses fallback defaults
    sb = StateBuilder(buf)
    result = sb.get_with_raw()

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_get_with_raw_normalised_matches_get():
    from thermalkit.mlx_wrap.state_builder import StateBuilder, FEATURE_DIM
    from thermalkit.sensor import SensorBuffer

    buf = SensorBuffer()
    sb = StateBuilder(buf)
    vec_direct = sb.get()
    vec_via_raw, _ = sb.get_with_raw()

    np.testing.assert_array_equal(vec_direct, vec_via_raw)


def test_get_with_raw_keys():
    from thermalkit.mlx_wrap.state_builder import StateBuilder
    from thermalkit.sensor import SensorBuffer

    buf = SensorBuffer()
    sb = StateBuilder(buf)
    _, raw = sb.get_with_raw()

    expected_keys = {
        "cpu_temp_c", "gpu_temp_c", "power_w",
        "mem_pressure_gb", "batt_pct", "hour_of_day",
    }
    assert expected_keys == set(raw.keys())


def test_get_with_raw_values_in_range():
    from thermalkit.mlx_wrap.state_builder import StateBuilder, FEATURE_DIM
    from thermalkit.sensor import SensorBuffer

    buf = SensorBuffer()
    sb = StateBuilder(buf)
    normed, raw = sb.get_with_raw()

    assert normed.shape == (FEATURE_DIM,)
    assert (normed >= 0.0).all() and (normed <= 1.0).all()
    assert raw["hour_of_day"] >= 0.0
    assert raw["batt_pct"] >= 0.0


# ---------------------------------------------------------------------------
# _print_explain_pre output structure
# ---------------------------------------------------------------------------

def _make_policy_with_updates(n=10):
    p = LinUCBPolicy(alpha=1.0)
    rng = np.random.default_rng(0)
    for _ in range(n):
        state = rng.random(6).astype(np.float32)
        p.update(state, 0, 80.0)
        p.update(state, 1, 100.0)
    return p


def _capture_explain_pre(policy=None, chosen_action=1):
    if policy is None:
        policy = _make_policy_with_updates()
    state = np.ones(6, dtype=np.float32) * 0.5
    raw = {
        "cpu_temp_c": 64.2, "gpu_temp_c": 0.0, "power_w": 0.0,
        "mem_pressure_gb": 8.1, "batt_pct": 71.0, "hour_of_day": 14.0,
    }
    buf = io.StringIO()
    with patch("builtins.print", side_effect=lambda *a, **kw: buf.write(str(a[0]) + "\n")):
        _print_explain_pre(state, raw, policy, chosen_action)
    return buf.getvalue()


def test_explain_pre_contains_separator():
    out = _capture_explain_pre()
    assert _SEP in out


def test_explain_pre_shows_state_vector_header():
    out = _capture_explain_pre()
    assert "State vector" in out


def test_explain_pre_shows_all_feature_names():
    out = _capture_explain_pre()
    for key in ["cpu_temp_c", "gpu_temp_c", "power_w", "mem_pressure_gb", "batt_pct", "hour_of_day"]:
        assert key in out, f"Missing feature: {key}"


def test_explain_pre_shows_ucb_scores():
    out = _capture_explain_pre()
    assert "UCB scores" in out
    for a in range(N_ACTIONS):
        assert f"batch_size={batch_size_for(a)}" in out


def test_explain_pre_marks_chosen_action():
    out = _capture_explain_pre(chosen_action=2)
    # The chosen action line must have the "← chosen" marker
    lines = out.split("\n")
    chosen_line = next((l for l in lines if "batch_size=4" in l), None)
    assert chosen_line is not None
    assert "chosen" in chosen_line


def test_explain_pre_shows_decision_and_reason():
    out = _capture_explain_pre()
    assert "Decision" in out
    assert "Reason" in out


def test_explain_pre_warmup_shown_correctly():
    # Fresh policy: warmup not done
    fresh = LinUCBPolicy()
    out = _capture_explain_pre(policy=fresh, chosen_action=0)
    assert "warmup" in out.lower()


def test_explain_pre_policy_call_count_shown():
    policy = _make_policy_with_updates(n=5)
    out = _capture_explain_pre(policy=policy)
    # call_count should appear somewhere
    assert str(policy.call_count) in out


# ---------------------------------------------------------------------------
# _print_explain_post output structure
# ---------------------------------------------------------------------------

def _capture_explain_post(tok_sec=118.3, tokens=150):
    buf = io.StringIO()
    with patch("builtins.print", side_effect=lambda *a, **kw: buf.write(str(a[0]) + "\n")):
        _print_explain_post(tok_sec, tokens)
    return buf.getvalue()


def test_explain_post_shows_result():
    out = _capture_explain_post(118.3, 150)
    assert "118.3" in out
    assert "150" in out


def test_explain_post_shows_tok_sec_label():
    out = _capture_explain_post()
    assert "tok/sec" in out


def test_explain_post_contains_separator():
    out = _capture_explain_post()
    assert _SEP in out


# ---------------------------------------------------------------------------
# ForgeDaemon.generate(explain=True) — integration (no real model)
# ---------------------------------------------------------------------------

def _make_mock_daemon(tmp_path):
    """Return a ForgeDaemon with mocked model/tokenizer and started sensors."""
    from thermalkit.daemon.daemon import ForgeDaemon

    # Build a minimal response mock that mlx_lm.stream_generate would yield
    mock_resp = MagicMock()
    mock_resp.text = "hello"
    mock_resp.generation_tps = 118.3
    mock_resp.generation_tokens = 5

    daemon = ForgeDaemon(
        model=MagicMock(),
        tokenizer=MagicMock(),
        policy_path=tmp_path / "policy.npz",
        db_path=tmp_path / "inference.db",
    )
    daemon.start()
    return daemon, mock_resp


@patch("thermalkit.daemon.daemon._print_explain_pre")
@patch("thermalkit.daemon.daemon._print_explain_post")
def test_explain_true_calls_both_helpers(mock_post, mock_pre, tmp_path):
    """With explain=True, both pre and post helpers must be called."""
    daemon, mock_resp = _make_mock_daemon(tmp_path)

    with patch("mlx_lm.stream_generate", return_value=iter([mock_resp])):
        daemon.generate("hello", explain=True, max_tokens=5)

    daemon.stop()
    mock_pre.assert_called_once()
    mock_post.assert_called_once()


@patch("thermalkit.daemon.daemon._print_explain_pre")
@patch("thermalkit.daemon.daemon._print_explain_post")
def test_explain_false_calls_no_helpers(mock_post, mock_pre, tmp_path):
    """With explain=False (default), neither helper should be called."""
    daemon, mock_resp = _make_mock_daemon(tmp_path)

    with patch("mlx_lm.stream_generate", return_value=iter([mock_resp])):
        daemon.generate("hello", max_tokens=5)

    daemon.stop()
    mock_pre.assert_not_called()
    mock_post.assert_not_called()


@patch("thermalkit.daemon.daemon._print_explain_pre")
@patch("thermalkit.daemon.daemon._print_explain_post")
def test_explain_returns_same_text(mock_post, mock_pre, tmp_path):
    """explain=True must not change the returned text."""
    daemon, mock_resp = _make_mock_daemon(tmp_path)

    with patch("mlx_lm.stream_generate", return_value=iter([mock_resp])):
        text = daemon.generate("hello", explain=True, max_tokens=5)

    daemon.stop()
    assert text == "hello"


def test_explain_pre_chosen_action_matches_policy_select(tmp_path):
    """The action printed by explain must match what select_action would return."""
    import numpy as np
    from thermalkit.daemon.daemon import _print_explain_pre
    from thermalkit.bandit.policy import LinUCBPolicy

    policy = LinUCBPolicy(alpha=1.0)
    state = np.ones(6, dtype=np.float32) * 0.5
    raw = {
        "cpu_temp_c": 55.0, "gpu_temp_c": 0.0, "power_w": 0.0,
        "mem_pressure_gb": 4.0, "batt_pct": 80.0, "hour_of_day": 10.0,
    }

    expected_action = policy.select_action(state)

    captured_actions = []
    original = _print_explain_pre

    def spy(s, r, p, action):
        captured_actions.append(action)
        # Don't actually print
    with patch("thermalkit.daemon.daemon._print_explain_pre", side_effect=spy):
        from thermalkit.daemon.daemon import ForgeDaemon
        daemon = ForgeDaemon(
            model=MagicMock(), tokenizer=MagicMock(),
            policy_path=tmp_path / "p.npz", db_path=tmp_path / "i.db",
        )
        daemon.start()
        mock_resp = MagicMock()
        mock_resp.text = "x"
        mock_resp.generation_tps = 100.0
        mock_resp.generation_tokens = 5
        with patch("mlx_lm.stream_generate", return_value=iter([mock_resp])):
            daemon.generate("hi", explain=True, max_tokens=5)
        daemon.stop()

    assert len(captured_actions) == 1
    assert captured_actions[0] == expected_action
