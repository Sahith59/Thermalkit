"""Feature 3 tests — thermal-penalty reward function."""

import os
import numpy as np
import pytest

from thermalkit.mlx_wrap.reward import compute_reward


# ---------------------------------------------------------------------------
# Core formula correctness
# ---------------------------------------------------------------------------

def test_no_penalty_below_threshold():
    """At or below 70°C, reward equals raw tok/sec (no penalty)."""
    assert compute_reward(100.0, 60.0) == pytest.approx(100.0)
    assert compute_reward(100.0, 70.0) == pytest.approx(100.0)


def test_penalty_above_threshold():
    """Above 70°C, reward is reduced."""
    r = compute_reward(100.0, 80.0)
    assert r < 100.0
    # Expected: 100 / (1 + 0.05 * 10) = 100 / 1.5 = 66.67
    assert r == pytest.approx(100.0 / 1.5, rel=1e-4)


def test_penalty_grows_with_temperature():
    """Hotter = lower reward for same tok/sec."""
    r70 = compute_reward(100.0, 70.0)
    r80 = compute_reward(100.0, 80.0)
    r90 = compute_reward(100.0, 90.0)
    assert r70 > r80 > r90


def test_reward_always_positive():
    """Reward must never be negative, even at extreme temperatures."""
    assert compute_reward(100.0, 200.0) > 0.0


def test_none_cpu_temp_passthrough():
    """When cpu_temp is None (sensor unavailable), return raw tok/sec."""
    assert compute_reward(100.0, None) == pytest.approx(100.0)


def test_zero_tok_sec():
    """Zero tok/sec returns zero regardless of temperature."""
    assert compute_reward(0.0, 80.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

def test_lambda_zero_disables_penalty(monkeypatch):
    """THERMALKIT_LAMBDA=0 disables the penalty entirely."""
    monkeypatch.setenv("THERMALKIT_LAMBDA", "0")
    assert compute_reward(100.0, 90.0) == pytest.approx(100.0)


def test_custom_lambda(monkeypatch):
    """Custom lambda changes the penalty slope."""
    monkeypatch.setenv("THERMALKIT_LAMBDA", "0.10")  # steeper penalty
    r_default = compute_reward.__wrapped__(100.0, 80.0) if hasattr(compute_reward, "__wrapped__") else None
    r_custom = compute_reward(100.0, 80.0)
    # At lambda=0.10, 10°C above threshold: 100 / (1 + 0.10*10) = 100/2 = 50
    assert r_custom == pytest.approx(100.0 / 2.0, rel=1e-4)


def test_custom_threshold(monkeypatch):
    """Custom threshold shifts where penalty starts."""
    monkeypatch.setenv("THERMALKIT_T_THRESH", "60.0")
    # At 70°C (10°C above new threshold): 100 / (1 + 0.05*10) = 66.67
    r = compute_reward(100.0, 70.0)
    assert r == pytest.approx(100.0 / 1.5, rel=1e-4)


# ---------------------------------------------------------------------------
# Integration: bandit policy converges differently with thermal reward
# ---------------------------------------------------------------------------

def test_thermal_reward_makes_policy_prefer_cooler_action():
    """With thermal penalty, bandit should learn to avoid hot actions.

    Scenario: two actions, same raw tok/sec, but action 0 runs at 80°C
    and action 1 runs at 60°C.  After enough updates, policy should
    prefer action 1 (cooler).
    """
    from thermalkit.bandit.policy import LinUCBPolicy
    import numpy as np

    rng = np.random.default_rng(42)
    policy = LinUCBPolicy(alpha=0.5)

    for _ in range(50):
        state = rng.random(6).astype(np.float32)

        # Feed action 0: hot (80°C), action 1: cool (60°C), same raw tok/sec
        policy.update(state, 0, compute_reward(100.0, 80.0))  # → 66.7
        policy.update(state, 1, compute_reward(100.0, 60.0))  # → 100.0

    state = np.ones(6, dtype=np.float32) * 0.5
    scores = policy.ucb_scores(state)
    # After training, action 1 (cool) must score higher than action 0 (hot)
    assert scores[1] > scores[0], (
        f"Policy should prefer cooler action 1 (score={scores[1]:.2f}) "
        f"over hotter action 0 (score={scores[0]:.2f})"
    )
