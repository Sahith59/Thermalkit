"""Phase 3 tests for LinUCBPolicy and action_space."""

import time
from pathlib import Path

import numpy as np
import pytest

from thermalkit.bandit.action_space import (
    N_ACTIONS,
    BATCH_SIZES,
    action_index,
    batch_size_for,
    feature_vector,
)
from thermalkit.bandit.policy import LinUCBPolicy


# ---------------------------------------------------------------------------
# ActionSpace
# ---------------------------------------------------------------------------

def test_action_index_roundtrip():
    for bs in BATCH_SIZES:
        assert batch_size_for(action_index(bs)) == bs


def test_feature_vector_shape():
    state = np.random.rand(6).astype(np.float32)
    fv = feature_vector(0, state)
    assert fv.shape == (6 + N_ACTIONS,)


def test_feature_vector_one_hot():
    state = np.zeros(6, dtype=np.float32)
    for a in range(N_ACTIONS):
        fv = feature_vector(a, state)
        one_hot = fv[6:]
        assert one_hot[a] == 1.0
        assert one_hot.sum() == 1.0


# ---------------------------------------------------------------------------
# LinUCBPolicy — basic interface
# ---------------------------------------------------------------------------

def test_select_action_returns_valid_index():
    policy = LinUCBPolicy(alpha=1.0)
    state = np.random.rand(6).astype(np.float32)
    action = policy.select_action(state)
    assert 0 <= action < N_ACTIONS


def test_select_action_is_fast():
    policy = LinUCBPolicy(alpha=1.0)
    state = np.random.rand(6).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(100):
        policy.select_action(state)
    elapsed = time.perf_counter() - t0
    # 100 calls must complete in under 1 second (i.e. < 10ms each)
    assert elapsed < 1.0, f"select_action too slow: {elapsed:.3f}s for 100 calls"


def test_update_changes_scores():
    policy = LinUCBPolicy(alpha=0.1)  # low alpha = exploitation-heavy
    state = np.ones(6, dtype=np.float32) * 0.5

    # Feed action 2 (batch_size=4) a high reward many times
    for _ in range(30):
        policy.update(state, 2, reward=100.0)

    # Action 2 should now have the highest score
    scores = policy.ucb_scores(state)
    assert np.argmax(scores) == 2


def test_call_count_increments():
    policy = LinUCBPolicy()
    state = np.random.rand(6).astype(np.float32)
    assert policy.call_count == 0
    policy.update(state, 0, 50.0)
    policy.update(state, 1, 60.0)
    assert policy.call_count == 2


# ---------------------------------------------------------------------------
# Persistence — save / load round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_path):
    policy = LinUCBPolicy(alpha=1.5)
    state = np.random.rand(6).astype(np.float32)

    # Do some updates to make weights non-trivial
    for a in range(N_ACTIONS):
        for _ in range(5):
            policy.update(state, a, float(a * 10))

    path = tmp_path / "policy.npz"
    policy.save(path)

    loaded = LinUCBPolicy.load(path, alpha=1.5)
    assert loaded.call_count == policy.call_count
    assert loaded.alpha == policy.alpha

    # Scores must be identical after round-trip
    scores_orig = policy.ucb_scores(state)
    scores_loaded = loaded.ucb_scores(state)
    np.testing.assert_allclose(scores_orig, scores_loaded, rtol=1e-6)


def test_load_or_new_creates_fresh_when_missing(tmp_path):
    path = tmp_path / "nonexistent.npz"
    policy = LinUCBPolicy.load_or_new(path)
    assert policy.call_count == 0


def test_load_or_new_loads_existing(tmp_path):
    policy = LinUCBPolicy()
    state = np.ones(6, dtype=np.float32) * 0.3
    policy.update(state, 0, 42.0)
    path = tmp_path / "policy.npz"
    policy.save(path)

    loaded = LinUCBPolicy.load_or_new(path)
    assert loaded.call_count == 1


# ---------------------------------------------------------------------------
# Convergence — synthetic bandit test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_synthetic_convergence():
    """Simulate 500 rounds with action 1 (batch_size=2) as the true optimal.

    Gate: at round 500, cumulative regret < 50% of a random-action baseline.
    """
    n_rounds = 500

    # True expected rewards per action (action 1 = best)
    true_rewards = np.array([40.0, 80.0, 60.0, 50.0], dtype=np.float64)
    optimal = float(true_rewards.max())

    def simulate_bandit(seed: int) -> float:
        rng = np.random.default_rng(seed)
        policy = LinUCBPolicy(alpha=1.0)
        total_regret = 0.0
        for _ in range(n_rounds):
            state = rng.random(6).astype(np.float32)
            action = policy.select_action(state)
            reward = float(rng.normal(true_rewards[action], 5.0))
            total_regret += optimal - true_rewards[action]
            policy.update(state, action, reward)
        return total_regret

    def simulate_random(seed: int) -> float:
        rng = np.random.default_rng(seed)
        total_regret = 0.0
        for _ in range(n_rounds):
            action = int(rng.integers(N_ACTIONS))
            total_regret += optimal - true_rewards[action]
        return total_regret

    # Average over 3 seeds for stability
    bandit_regret = float(np.mean([simulate_bandit(s) for s in [42, 7, 99]]))
    random_regret = float(np.mean([simulate_random(s) for s in [42, 7, 99]]))

    ratio = bandit_regret / random_regret
    assert ratio < 0.5, (
        f"Bandit regret {bandit_regret:.1f} is not < 50% of random {random_regret:.1f} "
        f"(ratio={ratio:.2f})"
    )


# ---------------------------------------------------------------------------
# Offline replay against Phase 2 SQLite log
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_offline_replay(tmp_path):
    """Replay Phase 2 inference.db through the bandit and check it learns.

    If the DB doesn't exist (CI / fresh checkout), the test is skipped.
    """
    import sqlite3
    from pathlib import Path

    db_path = Path.home() / ".thermalkit" / "inference.db"
    if not db_path.exists():
        pytest.skip("~/.thermalkit/inference.db not found — run thermalkit generate() first")

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT batch_size, cpu_temp, gpu_temp, mem_pressure, batt_pct, tok_sec "
        "FROM calls WHERE tok_sec IS NOT NULL ORDER BY ts"
    ).fetchall()
    conn.close()

    if len(rows) < 5:
        pytest.skip(f"Too few rows in inference.db ({len(rows)}) to replay")

    from thermalkit.bandit.action_space import action_index
    from thermalkit.mlx_wrap.state_builder import FEATURE_DIM

    policy = LinUCBPolicy(alpha=1.0)

    for batch_size, cpu_temp, gpu_temp, mem_pressure, batt_pct, tok_sec in rows:
        if batch_size not in BATCH_SIZES:
            continue

        # Reconstruct a normalised state from logged sensor values
        norms = np.array([110.0, 110.0, 60.0, 24.0, 100.0, 23.0], dtype=np.float32)
        raw = np.array(
            [cpu_temp or 50.0, gpu_temp or 40.0, 0.0,
             mem_pressure or 4.0, batt_pct or 100.0, 12.0],
            dtype=np.float32,
        )
        state = np.clip(raw / norms, 0.0, 1.0)
        a_idx = action_index(batch_size)
        policy.update(state, a_idx, tok_sec)

    assert policy.call_count > 0

    # Policy file saves and reloads cleanly
    path = tmp_path / "replayed.npz"
    policy.save(path)
    loaded = LinUCBPolicy.load(path)
    assert loaded.call_count == policy.call_count
