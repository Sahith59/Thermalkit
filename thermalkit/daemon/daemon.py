"""ForgeDaemon — closed-loop inference with LinUCB policy.

The daemon owns a model, a sensor buffer, a bandit policy, and an inference log.
On each call to generate(), it:

    1. Reads the current thermal state from SensorBuffer
    2. Asks LinUCBPolicy to pick a batch_size
    3. Runs inference with that batch_size
    4. Records tok/sec in TokSecMeter
    5. When enough reward history accumulates, calls policy.update()
    6. Saves the policy every SAVE_EVERY updates

The daemon can be used as a context manager:

    with ForgeDaemon.from_model("mlx-community/...") as daemon:
        text = daemon.generate("Your prompt here")
"""

from __future__ import annotations

import signal
import threading
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm
import numpy as np

from thermalkit.bandit.policy import LinUCBPolicy
from thermalkit.mlx_wrap._generate import _VALID_BATCH_SIZES
from thermalkit.mlx_wrap.inference_log import InferenceLog
from thermalkit.mlx_wrap.state_builder import StateBuilder
from thermalkit.mlx_wrap.tok_sec_meter import TokSecMeter
from thermalkit.sensor import SensorBuffer

_DEFAULT_POLICY_PATH = Path.home() / ".thermalkit" / "policy.npz"
_DEFAULT_DB_PATH     = Path.home() / ".thermalkit" / "inference.db"
_SAVE_EVERY = 10  # save policy every N updates


class ForgeDaemon:
    """Closed-loop inference daemon.

    Args:
        model:       Loaded MLX model (from mlx_lm.load).
        tokenizer:   Matching tokenizer.
        policy_path: Where to persist/load the LinUCB policy weights.
        db_path:     SQLite path for inference log.
        alpha:       LinUCB exploration coefficient.
    """

    def __init__(
        self,
        model,
        tokenizer,
        policy_path: Path = _DEFAULT_POLICY_PATH,
        db_path: Path = _DEFAULT_DB_PATH,
        alpha: float = 1.0,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._policy_path = Path(policy_path)
        self._alpha = alpha

        self._sensor = SensorBuffer()
        self._meter = TokSecMeter()
        self._log = InferenceLog(db_path=Path(db_path))
        self._policy = LinUCBPolicy.load_or_new(self._policy_path, alpha=alpha)

        self._lock = threading.Lock()
        self._updates_since_save = 0
        self._running = False

        # Track last state/action/tok_sec/cpu_temp for deferred reward update
        self._last_state: np.ndarray | None = None
        self._last_action: int | None = None
        self._last_tok_sec: float | None = None
        self._last_cpu_temp: float | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start sensor polling and log writer."""
        self._sensor.start()
        self._log.start()
        self._running = True

        # Register SIGTERM handler for clean shutdown
        signal.signal(signal.SIGTERM, lambda s, f: self.stop())

    def stop(self) -> None:
        """Flush log, save policy, stop sensors."""
        if not self._running:
            return
        self._running = False
        self._log.flush(timeout=3.0)
        self._save_policy()
        self._sensor.stop()

    def __enter__(self) -> "ForgeDaemon":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Core: generate
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        verbose: bool = False,
        explain: bool = False,
        **kwargs,
    ) -> str:
        """Run inference with bandit-chosen batch_size.

        Args:
            explain: If True, print a decision trace showing the state vector,
                     UCB scores, chosen action, and result for this call.

        Returns the generated text (same as mlx_lm.generate).
        """
        import time as _time

        # 1. Read sensor state (with raw values for explain mode)
        state_builder = StateBuilder(self._sensor)
        state, raw_values = state_builder.get_with_raw()

        # 2. Bandit picks action
        action_idx = self._policy.select_action(state)
        from thermalkit.bandit.action_space import batch_size_for, N_ACTIONS, BATCH_SIZES
        batch_size = batch_size_for(action_idx)

        # 2b. Print decision trace BEFORE inference
        if explain:
            _print_explain_pre(state, raw_values, self._policy, action_idx)

        # 3. Update previous round's reward if available
        #    (reward for call N is measured after call N completes,
        #     so we update the bandit at the start of call N+1)
        self._maybe_update_policy(state)

        # 4. Run inference
        sensor_state = self._sensor.get_current_state(timeout=0.5) or {}
        mx.set_default_device(mx.Device(mx.gpu))

        text = ""
        last_response = None
        for response in mlx_lm.stream_generate(
            self._model, self._tokenizer, prompt, **kwargs
        ):
            if verbose:
                print(response.text, end="", flush=True)
            text += response.text
            last_response = response

        if verbose:
            print()

        tok_sec = last_response.generation_tps if last_response else 0.0
        tokens_generated = last_response.generation_tokens if last_response else 0

        # 5. Record in meter
        self._meter.record_tps(tok_sec)
        reward = self._meter.get_reward()

        # 5b. Print post-inference result line
        if explain:
            _print_explain_post(tok_sec, tokens_generated)

        # 6. Store state+action+tok_sec+cpu_temp for next round's update
        with self._lock:
            self._last_state = state
            self._last_action = action_idx
            self._last_tok_sec = tok_sec if tok_sec > 0 else None
            self._last_cpu_temp = sensor_state.get("cpu_temp_c")

        # 7. Log to SQLite
        self._log.log({
            "ts": time.time(),
            "cpu_temp": sensor_state.get("cpu_temp_c"),
            "gpu_temp": sensor_state.get("gpu_temp_c"),
            "power_w": sensor_state.get("power_w"),
            "mem_pressure": sensor_state.get("mem_pressure_gb"),
            "batt_pct": sensor_state.get("batt_pct"),
            "compute_unit": "gpu",
            "batch_size": batch_size,
            "tokens_generated": tokens_generated,
            "tok_sec": tok_sec,
            "reward": reward,
        })

        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_update_policy(self, current_state: np.ndarray) -> None:
        """Update policy with reward from the previous call, if available.

        Preference order:
          1. Rolling 30s mean tok/sec (best signal, available after ~10s of calls)
          2. Last instantaneous tok/sec (immediate fallback so learning starts right away)
        """
        with self._lock:
            last_state    = self._last_state
            last_action   = self._last_action
            last_tok_sec  = self._last_tok_sec
            last_cpu_temp = self._last_cpu_temp

        if last_state is None or last_action is None:
            return

        # Use rolling reward when available; fall back to last call's tok/sec
        raw_reward = self._meter.get_reward()
        if raw_reward is None:
            raw_reward = last_tok_sec
        if raw_reward is None:
            return

        # Apply thermal penalty so the bandit prefers cooler operation
        from thermalkit.mlx_wrap.reward import compute_reward
        reward = compute_reward(raw_reward, last_cpu_temp)

        self._policy.update(last_state, last_action, reward)
        self._updates_since_save += 1

        if self._updates_since_save >= _SAVE_EVERY:
            self._save_policy()
            self._updates_since_save = 0

    def _save_policy(self) -> None:
        try:
            self._policy.save(self._policy_path)
        except Exception:
            pass  # never crash on save failure

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def call_count(self) -> int:
        return self._policy.call_count

    def status(self) -> dict:
        """Return a snapshot of daemon state for CLI display."""
        from thermalkit.bandit.action_space import batch_size_for, N_ACTIONS
        sensor_state = self._sensor.get_current_state(timeout=0.5) or {}
        state_builder = StateBuilder(self._sensor)
        state = state_builder.get()
        scores = self._policy.ucb_scores(state)
        best_action = int(np.argmax(scores))

        return {
            "running": self._running,
            "policy_calls": self._policy.call_count,
            "warmup_done": self._policy._warmup_done,
            "last_reward": self._meter.get_reward(),
            "cpu_temp_c": sensor_state.get("cpu_temp_c"),
            "gpu_temp_c": sensor_state.get("gpu_temp_c"),
            "batt_pct": sensor_state.get("batt_pct"),
            "mem_pressure_gb": sensor_state.get("mem_pressure_gb"),
            "best_action_batch_size": batch_size_for(best_action),
            "ucb_scores": {
                str(batch_size_for(a)): round(float(scores[a]), 4)
                for a in range(N_ACTIONS)
            },
            "policy_path": str(self._policy_path),
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_model(
        cls,
        model_path: str,
        policy_path: Path = _DEFAULT_POLICY_PATH,
        db_path: Path = _DEFAULT_DB_PATH,
        alpha: float = 1.0,
    ) -> "ForgeDaemon":
        """Load model + tokenizer then construct a ForgeDaemon."""
        model, tokenizer = mlx_lm.load(model_path)
        return cls(
            model=model,
            tokenizer=tokenizer,
            policy_path=policy_path,
            db_path=db_path,
            alpha=alpha,
        )


# ---------------------------------------------------------------------------
# Explain-mode helpers (module-level so they're easy to test independently)
# ---------------------------------------------------------------------------

_FEATURE_LABELS = [
    ("cpu_temp_c",      "{:.1f}°C",  ""),
    ("gpu_temp_c",      "{:.1f}°C",  "  (0°C = idle GPU, sensor activates under load)"),
    ("power_w",         "{:.1f}W",   "  (0.0 = powermetrics not configured)"),
    ("mem_pressure_gb", "{:.2f}GB",  ""),
    ("batt_pct",        "{:.0f}%",   ""),
    ("hour_of_day",     "{:.0f}h",   ""),
]

_SEP = "─" * 52


def _print_explain_pre(
    state: "np.ndarray",
    raw: dict,
    policy: "LinUCBPolicy",
    chosen_action: int,
) -> None:
    from thermalkit.bandit.action_space import N_ACTIONS, batch_size_for
    import numpy as np

    scores = policy.ucb_scores(state)

    print(f"\n[thermalkit] {_SEP}", flush=True)
    print("[thermalkit] Decision Trace", flush=True)
    print(f"[thermalkit] {_SEP}", flush=True)

    print("[thermalkit] State vector:", flush=True)
    keys = list(raw.keys())
    for i, (key, fmt, note) in enumerate(_FEATURE_LABELS):
        raw_val = raw.get(key, 0.0)
        norm_val = float(state[i])
        raw_str = fmt.format(raw_val)
        print(
            f"[thermalkit]   {key:<18}: {raw_str:<10} → {norm_val:.3f}{note}",
            flush=True,
        )

    warmup_str = "warmup complete" if policy._warmup_done else "still in warmup"
    print(
        f"\n[thermalkit] UCB scores ({policy.call_count} prior updates, {warmup_str}):",
        flush=True,
    )
    for a in range(N_ACTIONS):
        bs = batch_size_for(a)
        marker = "  ← chosen" if a == chosen_action else ""
        print(
            f"[thermalkit]   batch_size={bs}  →  {scores[a]:.3f}{marker}",
            flush=True,
        )

    reason = (
        "UCB winner, warmup complete"
        if policy._warmup_done
        else f"warmup round {policy.call_count + 1}/{N_ACTIONS}"
    )
    print(
        f"\n[thermalkit] Decision : batch_size={batch_size_for(chosen_action)}",
        flush=True,
    )
    print(f"[thermalkit] Reason   : {reason}", flush=True)
    print(f"[thermalkit] {_SEP}\n", flush=True)


def _print_explain_post(tok_sec: float, tokens_generated: int) -> None:
    print(
        f"\n[thermalkit] {_SEP}",
        flush=True,
    )
    print(
        f"[thermalkit] Result: {tok_sec:.1f} tok/sec  |  {tokens_generated} tokens generated",
        flush=True,
    )
    print(f"[thermalkit] {_SEP}\n", flush=True)
