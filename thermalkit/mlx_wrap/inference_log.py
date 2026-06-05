import queue
import sqlite3
import threading
import time
from pathlib import Path

_DEFAULT_DB = Path.home() / ".thermalkit" / "inference.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               REAL NOT NULL,
    cpu_temp         REAL,
    gpu_temp         REAL,
    power_w          REAL,
    mem_pressure     REAL,
    batt_pct         REAL,
    compute_unit     TEXT,
    batch_size       INTEGER,
    tokens_generated INTEGER,
    tok_sec          REAL,
    reward           REAL
);
"""

_INSERT = """
INSERT INTO calls
    (ts, cpu_temp, gpu_temp, power_w, mem_pressure, batt_pct,
     compute_unit, batch_size, tokens_generated, tok_sec, reward)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class InferenceLog:
    def __init__(self, db_path: Path = _DEFAULT_DB):
        self._db_path = db_path
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._started = False

    def start(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.execute(_SCHEMA)
        conn.commit()
        conn.close()
        self._started = True
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    def log(self, record: dict) -> None:
        if not self._started:
            self.start()
        self._queue.put(record)

    def flush(self, timeout: float = 2.0) -> None:
        """Block until the write queue is empty (used in tests and shutdown)."""
        deadline = time.monotonic() + timeout
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.05)

    def _writer_loop(self) -> None:
        conn = sqlite3.connect(str(self._db_path))
        while True:
            try:
                record = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                conn.execute(
                    _INSERT,
                    (
                        record.get("ts", time.time()),
                        record.get("cpu_temp"),
                        record.get("gpu_temp"),
                        record.get("power_w"),
                        record.get("mem_pressure"),
                        record.get("batt_pct"),
                        record.get("compute_unit"),
                        record.get("batch_size"),
                        record.get("tokens_generated"),
                        record.get("tok_sec"),
                        record.get("reward"),
                    ),
                )
                conn.commit()
            except Exception:
                pass
