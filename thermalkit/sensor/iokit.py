import json
import os
import subprocess
import threading
import time
from pathlib import Path

def _find_binary() -> Path:
    # 1. Relative to this source file (works in dev/editable install)
    candidate = Path(__file__).parent / "iokit_reader" / "iokit_reader"
    if candidate.exists():
        return candidate
    # 2. FORGE_PROJECT_ROOT env var (set by activate script or explicitly)
    root = os.environ.get("FORGE_PROJECT_ROOT")
    if root:
        candidate = Path(root) / "thermalkit" / "sensor" / "iokit_reader" / "iokit_reader"
        if candidate.exists():
            return candidate
    # 3. Walk up from __file__ looking for pyproject.toml (project root marker)
    for parent in Path(__file__).parents:
        candidate = parent / "thermalkit" / "sensor" / "iokit_reader" / "iokit_reader"
        if candidate.exists():
            return candidate
    return Path(__file__).parent / "iokit_reader" / "iokit_reader"  # let start() raise

_BINARY = _find_binary()


class IOKitReader:
    """Subprocess wrapper for the Swift IOKit thermal sensor binary.

    Spawns the binary and reads JSONL from its stdout in a background thread.
    Automatically restarts on crash up to _MAX_RESTARTS times.
    """

    _MAX_RESTARTS = 5

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._running = False
        self._restarts = 0

    def start(self) -> None:
        if not _BINARY.exists():
            raise FileNotFoundError(
                f"IOKit reader binary not found at {_BINARY}. "
                "Run: swiftc main.swift -o iokit_reader inside forge/sensor/iokit_reader/"
            )
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def latest(self) -> dict | None:
        with self._lock:
            return self._latest.copy() if self._latest else None

    def _read_loop(self) -> None:
        while self._running and self._restarts <= self._MAX_RESTARTS:
            try:
                self._proc = subprocess.Popen(
                    [str(_BINARY)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
                for line in self._proc.stdout:  # type: ignore[union-attr]
                    if not self._running:
                        break
                    try:
                        reading = json.loads(line.strip())
                        with self._lock:
                            self._latest = reading
                    except json.JSONDecodeError:
                        pass
            except Exception:
                pass

            if self._running:
                self._restarts += 1
                time.sleep(1)
