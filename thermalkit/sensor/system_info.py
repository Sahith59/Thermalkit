import re
import subprocess
import threading
import time


class SystemInfoReader:
    """Polls battery % and memory pressure without sudo.

    - Battery: parsed from `pmset -g batt`
    - Memory pressure: computed from `vm_stat` + `sysctl hw.memsize`

    Updates every 5 seconds (these change slowly; no need for 1 Hz).
    """

    _POLL_INTERVAL = 5.0

    def __init__(self):
        self._lock = threading.Lock()
        self._batt_pct: float = 100.0
        self._mem_pressure_gb: float = 0.0
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._poll()  # populate immediately before returning
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def latest(self) -> dict:
        with self._lock:
            return {
                "batt_pct": self._batt_pct,
                "mem_pressure_gb": self._mem_pressure_gb,
            }

    def _loop(self) -> None:
        while self._running:
            time.sleep(self._POLL_INTERVAL)
            self._poll()

    def _poll(self) -> None:
        batt = _read_battery_pct()
        mem = _read_mem_pressure_gb()
        with self._lock:
            if batt is not None:
                self._batt_pct = batt
            if mem is not None:
                self._mem_pressure_gb = mem


def _read_battery_pct() -> float | None:
    """Parse battery % from pmset -g batt."""
    try:
        out = subprocess.check_output(
            ["pmset", "-g", "batt"], text=True, timeout=5
        )
        m = re.search(r"(\d+)%", out)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def _read_mem_pressure_gb() -> float | None:
    """Compute memory under pressure (active + wired pages) in GB from vm_stat."""
    try:
        # Page size
        ps_out = subprocess.check_output(
            ["sysctl", "-n", "hw.pagesize"], text=True, timeout=5
        )
        page_size = int(ps_out.strip())

        vm = subprocess.check_output(["vm_stat"], text=True, timeout=5)
        def _pages(label: str) -> int:
            m = re.search(rf"{label}:\s+([\d]+)", vm)
            return int(m.group(1)) if m else 0

        active = _pages("Pages active")
        wired  = _pages("Pages wired down")
        used_bytes = (active + wired) * page_size
        return round(used_bytes / 1024**3, 2)
    except Exception:
        return None
