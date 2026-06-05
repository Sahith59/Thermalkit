"""Disjoint LinUCB contextual bandit.

Each action has its own (A, b) parameter pair.  The UCB score for action a is:

    score_a = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)

where:
    x      = state vector (shape: state_dim)
    theta_a = A_a^{-1} b_a  (estimated reward weights)
    alpha  = exploration parameter (higher = more exploration)

On each update:
    A_a  += x x^T
    b_a  += reward * x
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from thermalkit.bandit.action_space import N_ACTIONS, BATCH_SIZES, batch_size_for
from thermalkit.mlx_wrap.state_builder import FEATURE_DIM

_STATE_DIM = FEATURE_DIM  # 6


class LinUCBPolicy:
    """Disjoint LinUCB with alpha-controlled exploration.

    Args:
        alpha:     Exploration coefficient. Higher = more exploration.
        state_dim: Dimensionality of the state vector (default 6).
    """

    def __init__(self, alpha: float = 1.0, state_dim: int = _STATE_DIM):
        self.alpha = alpha
        self.state_dim = state_dim
        self._call_count = 0

        # One (A, b) pair per action — A is (d x d), b is (d,)
        self._A = [np.eye(state_dim, dtype=np.float64) for _ in range(N_ACTIONS)]
        self._b = [np.zeros(state_dim, dtype=np.float64) for _ in range(N_ACTIONS)]

        # Warm-up: cycle through actions sequentially until every arm has been
        # updated at least once.  Without this, argmax ties on round 0 always
        # pick action 0, which then dominates UCB scores and blocks exploration.
        self._warmup_done = False

    # ------------------------------------------------------------------
    # Core bandit interface
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """Return the action index with the highest UCB score.

        For the first N_ACTIONS calls, cycles through actions sequentially so
        every arm gets at least one update before UCB scores are trusted.

        Args:
            state: Normalised state vector of shape (state_dim,).

        Returns:
            Integer in [0, N_ACTIONS).
        """
        if not self._warmup_done:
            # Round-robin warm-up: action index = call_count mod N_ACTIONS
            return self._call_count % N_ACTIONS

        x = state.astype(np.float64)
        scores = np.empty(N_ACTIONS)
        for a in range(N_ACTIONS):
            A_inv = np.linalg.inv(self._A[a])
            theta = A_inv @ self._b[a]
            uncertainty = float(np.sqrt(x @ A_inv @ x))
            scores[a] = float(theta @ x) + self.alpha * uncertainty
        return int(np.argmax(scores))

    def update(self, state: np.ndarray, action_idx: int, reward: float) -> None:
        """Update the model for the chosen action.

        Args:
            state:      State vector at decision time.
            action_idx: Index of the action that was taken.
            reward:     Observed reward (tok/sec from the 30s window).
        """
        x = state.astype(np.float64)
        self._A[action_idx] += np.outer(x, x)
        self._b[action_idx] += reward * x
        self._call_count += 1
        if not self._warmup_done and self._call_count >= N_ACTIONS:
            self._warmup_done = True

    def ucb_scores(self, state: np.ndarray) -> np.ndarray:
        """Return raw UCB scores for all actions. Useful for logging/debugging."""
        x = state.astype(np.float64)
        scores = np.empty(N_ACTIONS)
        for a in range(N_ACTIONS):
            A_inv = np.linalg.inv(self._A[a])
            theta = A_inv @ self._b[a]
            scores[a] = float(theta @ x) + self.alpha * float(np.sqrt(x @ A_inv @ x))
        return scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist policy weights to a .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "call_count": np.array(self._call_count),
            "warmup_done": np.array(self._warmup_done),
        }
        for a in range(N_ACTIONS):
            arrays[f"A_{a}"] = self._A[a]
            arrays[f"b_{a}"] = self._b[a]
        np.savez(str(path), **arrays)

    @classmethod
    def load(cls, path: str | Path, alpha: float = 1.0) -> "LinUCBPolicy":
        """Load policy weights from a .npz file."""
        path = Path(path)
        data = np.load(str(path))
        policy = cls(alpha=alpha)
        policy._call_count = int(data["call_count"])
        policy._warmup_done = bool(data["warmup_done"]) if "warmup_done" in data else (policy._call_count >= N_ACTIONS)
        for a in range(N_ACTIONS):
            policy._A[a] = data[f"A_{a}"]
            policy._b[a] = data[f"b_{a}"]
        return policy

    @classmethod
    def load_or_new(cls, path: str | Path, alpha: float = 1.0) -> "LinUCBPolicy":
        """Load from path if it exists, otherwise return a fresh policy."""
        path = Path(path)
        if path.exists():
            return cls.load(path, alpha=alpha)
        return cls(alpha=alpha)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        return self._call_count

    def best_action(self, state: np.ndarray) -> int:
        """Greedy action (alpha=0) — exploitation only. Used in evaluation."""
        x = state.astype(np.float64)
        scores = np.array([
            float((np.linalg.inv(self._A[a]) @ self._b[a]) @ x)
            for a in range(N_ACTIONS)
        ])
        return int(np.argmax(scores))

    def __repr__(self) -> str:
        return (
            f"LinUCBPolicy(alpha={self.alpha}, state_dim={self.state_dim}, "
            f"n_actions={N_ACTIONS}, call_count={self._call_count})"
        )
