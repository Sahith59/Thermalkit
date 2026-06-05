import time

import numpy as np

from thermalkit.sensor import SensorBuffer

# Normalisation denominators — keep aligned with PLAN.md state vector
_NORMS = np.array([110.0, 110.0, 60.0, 24.0, 100.0, 23.0], dtype=np.float32)

FEATURE_DIM = 6


class StateBuilder:
    """Converts the latest SensorBuffer reading into a normalised feature vector.

    Vector layout (all values clipped to [0, 1]):
        [cpu_temp, gpu_temp, power_w, mem_pressure_gb, batt_pct, hour_of_day]

    power_w is 0.0 until powermetrics integration lands in a later phase.
    """

    def __init__(self, sensor_buffer: SensorBuffer):
        self._buf = sensor_buffer

    def get(self) -> np.ndarray:
        """Return state vector of shape (6,). Falls back to safe defaults on timeout."""
        return self.get_with_raw()[0]

    def get_with_raw(self) -> tuple[np.ndarray, dict]:
        """Return (normalised_vector, raw_values_dict).

        raw_values_dict keys:
            cpu_temp_c, gpu_temp_c, power_w, mem_pressure_gb, batt_pct, hour_of_day
        """
        state = self._buf.get_current_state(timeout=0.5) or {}
        hour = float(time.localtime().tm_hour)
        raw_values = {
            "cpu_temp_c":      state.get("cpu_temp_c", 50.0),
            "gpu_temp_c":      state.get("gpu_temp_c", 40.0),
            "power_w":         state.get("power_w", 0.0),
            "mem_pressure_gb": state.get("mem_pressure_gb", 4.0),
            "batt_pct":        state.get("batt_pct", 100.0),
            "hour_of_day":     hour,
        }
        raw = np.array(list(raw_values.values()), dtype=np.float32)
        normalised = np.clip(raw / _NORMS, 0.0, 1.0)
        return normalised, raw_values
