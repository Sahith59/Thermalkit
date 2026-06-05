"""Thermal-penalty reward function.

Raw tok/sec maximises burst throughput.  This function teaches the bandit to
prefer configurations that sustain throughput without pushing the die
temperature — because a machine running at 78°C is one hard prompt away from
the OS thermal governor cutting clock speeds in half.

Formula:
    reward = tok_sec / (1.0 + lambda * max(0, cpu_temp - T_THRESHOLD))

Above T_THRESHOLD each additional degree costs `lambda * tok_sec` reward.
At lambda=0.05 and T_THRESHOLD=70°C:
    - 100 tok/s at 65°C  →  100.0  (no penalty, below threshold)
    - 100 tok/s at 75°C  →  100 / 1.25 = 80.0
    - 100 tok/s at 80°C  →  100 / 1.50 = 66.7

The penalty grows linearly, giving the bandit a smooth gradient to follow.

Tuning:
    THERMALKIT_LAMBDA   — float, default 0.05
    THERMALKIT_T_THRESH — float (°C), default 70.0
    Set THERMALKIT_LAMBDA=0 to disable the penalty entirely.
"""

import os

_DEFAULT_LAMBDA    = 0.05
_DEFAULT_THRESHOLD = 70.0


def compute_reward(tok_sec: float, cpu_temp: float | None) -> float:
    """Return thermally-penalised reward.

    Args:
        tok_sec:   Tokens per second from this inference call.
        cpu_temp:  CPU die temperature in °C, or None if unavailable.

    Returns:
        Adjusted reward as a float.  Always >= 0.
    """
    lam   = float(os.environ.get("THERMALKIT_LAMBDA",    _DEFAULT_LAMBDA))
    t_thr = float(os.environ.get("THERMALKIT_T_THRESH",  _DEFAULT_THRESHOLD))

    if lam == 0.0 or cpu_temp is None:
        return float(tok_sec)

    excess = max(0.0, cpu_temp - t_thr)
    penalty = 1.0 + lam * excess
    return float(tok_sec) / penalty
