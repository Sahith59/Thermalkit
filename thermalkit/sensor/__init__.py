import threading
import time
from collections import deque

from .iokit import IOKitReader
from .system_info import SystemInfoReader


class SensorBuffer:
    """Fuses IOKit thermal readings + system info into a thread-safe ring buffer.

    Holds up to 60 seconds of readings (maxlen=60 at 1 Hz).
    Call start() before use; call stop() to clean up.
    """

    _MAXLEN = 60

    def __init__(self):
        self._iokit = IOKitReader()
        self._sysinfo = SystemInfoReader()
        self._buf: deque[dict] = deque(maxlen=self._MAXLEN)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._iokit.start()
        self._sysinfo.start()
        self._running = True
        self._thread = threading.Thread(target=self._fuse_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._iokit.stop()
        self._sysinfo.stop()

    def get_current_state(self, timeout: float = 2.0) -> dict | None:
        """Return the most recent fused reading. Blocks up to `timeout` seconds
        waiting for the first sample if the buffer is empty."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._buf:
                    return dict(self._buf[-1])
            time.sleep(0.05)
        return None

    def history(self, n: int | None = None) -> list[dict]:
        """Return up to n most-recent readings (default: all in buffer)."""
        with self._lock:
            buf = list(self._buf)
        return buf if n is None else buf[-n:]

    def _fuse_loop(self) -> None:
        while self._running:
            start = time.monotonic()

            iokit_data  = self._iokit.latest()
            sys_data    = self._sysinfo.latest()

            if iokit_data:
                reading = {
                    "ts":              iokit_data.get("ts", time.time()),
                    "cpu_temp_c":      iokit_data.get("cpu_temp_c", 0.0),
                    "gpu_temp_c":      iokit_data.get("gpu_temp_c", 0.0),
                    "batt_pct":        sys_data.get("batt_pct", 100.0),
                    "mem_pressure_gb": sys_data.get("mem_pressure_gb", 0.0),
                }
                with self._lock:
                    self._buf.append(reading)

            elapsed = time.monotonic() - start
            time.sleep(max(0.0, 1.0 - elapsed))
