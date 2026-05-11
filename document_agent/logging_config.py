from __future__ import annotations

import collections
import datetime as dt
import logging
import sys
import threading
from typing import List, Optional


class RingBufferHandler(logging.Handler):
    """Thread-safe fixed-capacity rotating log buffer for in-UI log viewing."""

    def __init__(self, capacity: int = 2000) -> None:
        super().__init__()
        self._buffer: collections.deque = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._seq = 0
        self._capacity = capacity

    def emit(self, record: logging.LogRecord) -> None:
        try:
            with self._lock:
                self._seq += 1
                self._buffer.append({
                    "seq": self._seq,
                    "ts": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(timespec="seconds"),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                })
        except Exception:
            self.handleError(record)

    def get_records(
        self,
        *,
        limit: int = 100,
        level: Optional[str] = None,
        q: Optional[str] = None,
        since_seq: int = 0,
    ) -> List[dict]:
        _order = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        min_lvl = _order.get((level or "").upper(), 0)
        q_lower = q.lower() if q else None
        with self._lock:
            snapshot = list(self._buffer)
        result = []
        for rec in reversed(snapshot):
            if rec["seq"] <= since_seq:
                continue
            if min_lvl and _order.get(rec["level"], 0) < min_lvl:
                continue
            if q_lower and q_lower not in rec["message"].lower() and q_lower not in rec["logger"].lower():
                continue
            result.append(rec)
            if len(result) >= limit:
                break
        return result

    def stats(self) -> dict:
        with self._lock:
            used = len(self._buffer)
            max_seq = self._seq
        return {"buffer_capacity": self._capacity, "buffer_used": used, "max_seq": max_seq}


_RING_BUFFER: Optional[RingBufferHandler] = None


def get_ring_buffer() -> RingBufferHandler:
    if _RING_BUFFER is None:
        raise RuntimeError("Ring buffer not initialized; call configure_logging() first.")
    return _RING_BUFFER


def configure_logging(level: str = "INFO") -> None:
    global _RING_BUFFER
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(
        logging.Formatter(
            '{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
        )
    )
    root.addHandler(stream_handler)

    ring = RingBufferHandler(capacity=2000)
    root.addHandler(ring)
    _RING_BUFFER = ring
