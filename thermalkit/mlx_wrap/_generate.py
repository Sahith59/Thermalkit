import importlib
import threading
import time

import mlx.core as mx
import mlx_lm

_mlx_gen = importlib.import_module("mlx_lm.generate")

from thermalkit.sensor import SensorBuffer
from thermalkit.mlx_wrap.inference_log import InferenceLog
from thermalkit.mlx_wrap.tok_sec_meter import TokSecMeter

# Module-level shared instances — lazily initialised on first call, thread-safe.
_lock = threading.Lock()
_sensor_buf: SensorBuffer | None = None
_meter = TokSecMeter()
_log: InferenceLog | None = None

_VALID_BATCH_SIZES = {1, 2, 4, 8}


def _sensor() -> SensorBuffer:
    global _sensor_buf
    if _sensor_buf is None:
        with _lock:
            if _sensor_buf is None:
                _sensor_buf = SensorBuffer()
                _sensor_buf.start()
    return _sensor_buf


def _inference_log() -> InferenceLog:
    global _log
    if _log is None:
        with _lock:
            if _log is None:
                _log = InferenceLog()
                _log.start()
    return _log


def generate(
    model,
    tokenizer,
    prompt: str,
    *,
    batch_size: int = 1,
    verbose: bool = False,
    **kwargs,
) -> str:
    """Drop-in replacement for mlx_lm.generate() with thermal-aware observability.

    Extra keyword arg:
        batch_size: 1 | 2 | 4 | 8 — the bandit's primary action.
                    Passed to mlx_lm as max_tokens multiplier proxy; logged for
                    bandit training.  Inference always uses the GPU (Metal) since
                    MLX model weights are loaded onto GPU at load time.

    Returns the same str as mlx_lm.generate().
    """
    if batch_size not in _VALID_BATCH_SIZES:
        raise ValueError(f"batch_size must be one of {_VALID_BATCH_SIZES}, got {batch_size}")

    # Always run on GPU — model weights are placed on GPU at load time.
    # set_default_device after load doesn't move weights, so CPU mode is a no-op.
    mx.set_default_device(mx.Device(mx.gpu))

    # Snapshot sensor state before inference starts
    buf = _sensor()
    sensor_state = buf.get_current_state(timeout=0.5) or {}

    # Stream inference so we can capture token-level stats from MLX
    t0 = time.perf_counter()
    text = ""
    last_response = None
    for response in mlx_lm.stream_generate(model, tokenizer, prompt, **kwargs):
        if verbose:
            print(response.text, end="", flush=True)
        text += response.text
        last_response = response

    if verbose:
        print()

    wall_time = time.perf_counter() - t0

    if last_response is not None:
        tokens_generated = last_response.generation_tokens
        tok_sec = last_response.generation_tps
    else:
        tokens_generated = 0
        tok_sec = 0.0

    _meter.record_tps(tok_sec)
    reward = _meter.get_reward()

    _inference_log().log(
        {
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
        }
    )

    return text
