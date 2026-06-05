import threading
import time
from collections import deque

_WINDOW_SEC = 30.0
_MIN_DATA_SEC = 10.0


class TokSecMeter:
    """Tracks instantaneous tok/sec per call and a rolling 30-second reward."""

    def __init__(self):
        self._lock = threading.Lock()
        # Each entry: (timestamp, tok_sec)
        self._entries: deque[tuple[float, float]] = deque()

    def record(self, tokens_generated: int, wall_time_sec: float) -> float:
        """Record a completed call. Returns instantaneous tok/sec."""
        tok_sec = tokens_generated / wall_time_sec if wall_time_sec > 0 else 0.0
        with self._lock:
            self._entries.append((time.time(), tok_sec))
            self._trim()
        return tok_sec

    def record_tps(self, tok_sec: float) -> None:
        """Record a tok/sec value directly (when MLX already computed it)."""
        with self._lock:
            self._entries.append((time.time(), tok_sec))
            self._trim()

    def get_reward(self) -> float | None:
        """Mean tok/sec over the last 30s. Returns None if < 10s of data."""
        now = time.time()
        with self._lock:
            self._trim()
            if not self._entries:
                return None
            span = now - self._entries[0][0]
            if span < _MIN_DATA_SEC:
                return None
            return sum(e[1] for e in self._entries) / len(self._entries)

    def _trim(self) -> None:
        cutoff = time.time() - _WINDOW_SEC
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()
